//! Dedicated SQLite worker. SQL never runs on a streaming or UI task.
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, mpsc};
use step_harness_providers::request_metrics::{
    ExecutionRecord, RequestContext, RequestObserver, RequestRecord,
};
use tokio::sync::oneshot;

type Job = Box<dyn FnOnce(Result<&mut Connection, String>) + Send>;
pub struct RequestStore {
    tx: Mutex<Option<mpsc::Sender<Job>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    error: Arc<Mutex<Option<String>>>,
    run_id: String,
    notify: Arc<dyn Fn(String) + Send + Sync>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cursor {
    pub started_at: u64,
    pub id: String,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordFilter {
    pub from_ms: Option<u64>,
    pub to_ms: Option<u64>,
    pub model: Option<String>,
    pub status: Option<String>,
    pub cursor: Option<Cursor>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordPage {
    pub records: Vec<RequestRecord>,
    pub next_cursor: Option<Cursor>,
    pub models: Vec<String>,
}
#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceSummary {
    pub last_round_tps: Option<f64>,
    pub session_tps: Option<f64>,
    pub last_round_samples: u64,
    pub session_samples: u64,
    pub last_round_usage: UsageCounters,
    pub session_usage: UsageCounters,
    pub session_rounds: u64,
}
#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageCounters {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}
#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HealthPoint {
    pub day: usize,
    pub provider: String,
    pub model: String,
    pub tps: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub tps_samples: u64,
    pub first_token_samples: u64,
    pub timestamp: Option<u64>,
    pub request_id: Option<String>,
    pub status: Option<String>,
}
fn initialize(path: &std::path::Path, run_id: &str) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let connection = Connection::open(path).map_err(|e| e.to_string())?;
    connection
        .busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| e.to_string())?;
    connection
        .execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")
        .map_err(|e| e.to_string())?;
    let version: u32 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;
    if version > 2 {
        return Err("请求记录数据库版本高于当前应用支持的版本".into());
    }
    if version == 0 {
        connection.execute_batch("BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, logical_id TEXT NOT NULL, attempt INTEGER NOT NULL,
                session_id TEXT NOT NULL, execution_id TEXT, purpose TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                started_at INTEGER NOT NULL, ended_at INTEGER, status TEXT NOT NULL,
                input_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, output_tokens INTEGER,
                first_token_ms REAL, generation_ms REAL, tps REAL, http_status INTEGER, error_kind TEXT, response_model TEXT
            );
            CREATE INDEX IF NOT EXISTS requests_time ON requests(started_at DESC,id DESC);
            CREATE INDEX IF NOT EXISTS requests_model_time ON requests(provider,model,started_at);
            CREATE INDEX IF NOT EXISTS requests_session_execution ON requests(session_id,execution_id,started_at);
            CREATE TABLE IF NOT EXISTS instances(id TEXT PRIMARY KEY,pid INTEGER NOT NULL,started_at INTEGER NOT NULL);
            PRAGMA user_version=1; COMMIT;").map_err(|e| e.to_string())?;
    }
    if version < 2 {
        // Add an empty execution ledger; do not import historical transcripts.
        connection.execute_batch("BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS executions (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, session_id TEXT NOT NULL,
                started_at INTEGER NOT NULL, ended_at INTEGER, duration_ms INTEGER, status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS executions_session_time ON executions(session_id,started_at);
            PRAGMA user_version=2; COMMIT;").map_err(|e| e.to_string())?;
    }
    let owners: Vec<(String, u32, i64)> = connection
        .prepare("SELECT id,pid,started_at FROM instances")
        .map_err(|e| e.to_string())?
        .query_map([], |r| Ok((r.get(0)?, r.get::<_, u32>(1)?, r.get(2)?)))
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    for (id, pid, started_at) in owners {
        // Permission errors cannot establish that another process has exited.
        let identity = process_identity(pid);
        if matches!(identity, Ok(None))
            || matches!(identity, Ok(Some(identity)) if started_at != 0 && identity != started_at)
        {
            connection.execute("UPDATE executions SET status='interrupted' WHERE run_id=?1 AND status='running'", [&id]).map_err(|e| e.to_string())?;
            connection
                .execute(
                    "UPDATE requests SET status='interrupted',error_kind='process_exit',ended_at=?1
                WHERE run_id=?2 AND status='running'",
                    params![integer(step_harness_providers::utils::now_ms())?, id],
                )
                .map_err(|e| e.to_string())?;
            connection
                .execute("DELETE FROM instances WHERE id=?1", [id])
                .map_err(|e| e.to_string())?;
        }
    }
    let process_started_at = process_identity(std::process::id())
        .ok()
        .flatten()
        .unwrap_or(0);
    connection
        .execute(
            "INSERT INTO instances VALUES (?1,?2,?3)",
            params![run_id, std::process::id(), process_started_at],
        )
        .map_err(|e| e.to_string())?;
    Ok(connection)
}
impl RequestStore {
    pub fn open(
        path: std::path::PathBuf,
        notify: impl Fn(String) + Send + Sync + 'static,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Job>();
        let error = Arc::new(Mutex::new(None));
        let worker_error = error.clone();
        let run_id = step_harness_providers::request_metrics::new_id();
        let worker_id = run_id.clone();
        let worker = std::thread::spawn(move || {
            let result = initialize(&path, &worker_id);
            let initialization_error = result.as_ref().err().cloned();
            *worker_error.lock().unwrap_or_else(|p| p.into_inner()) = initialization_error.clone();
            let mut connection = result.ok();
            for job in rx {
                job(connection.as_mut().ok_or_else(|| {
                    initialization_error
                        .clone()
                        .unwrap_or_else(|| "请求记录存储不可用".into())
                }));
            }
            if let Some(connection) = connection {
                let _ = connection.execute_batch("PRAGMA wal_checkpoint(PASSIVE)");
            }
        });
        Arc::new(Self {
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
            error,
            run_id,
            notify: Arc::new(notify),
        })
    }
    async fn query<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Connection) -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let (tx, rx) = oneshot::channel();
        let job: Job = Box::new(move |connection| {
            let _ = tx.send(connection.and_then(operation));
        });
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .ok_or("请求记录存储已关闭")?
            .send(job)
            .map_err(|_| "请求记录线程已停止")?;
        rx.await.map_err(|_| "请求记录查询已取消".to_string())?
    }
    pub fn shutdown(&self) {
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(worker) = self.worker.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = worker.join();
        }
    }
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    pub async fn usage(&self) -> Result<crate::usage::UsageSnapshot, String> {
        self.query(|connection| {
            let mut statement = connection.prepare("SELECT started_at,provider,model,
                COALESCE(input_tokens,0)+COALESCE(output_tokens,0) FROM requests ORDER BY started_at,id").map_err(|e| e.to_string())?;
            let records = statement.query_map([], |row| Ok(crate::usage::UsageRecord {
                timestamp: row.get::<_,i64>(0)? as u64, provider: row.get(1)?, model: row.get(2)?, tokens: row.get::<_,i64>(3)? as u64,
            })).map_err(|e| e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e| e.to_string())?;
            let longest_chat_ms = connection.query_row("SELECT COALESCE(MAX(total),0) FROM
                (SELECT SUM(duration_ms) AS total FROM executions WHERE duration_ms IS NOT NULL GROUP BY session_id)", [], |row| row.get::<_,i64>(0)).map_err(|e| e.to_string())? as u64;
            let unknown_token_requests = connection.query_row("SELECT COUNT(*) FROM requests WHERE input_tokens IS NULL OR output_tokens IS NULL", [], |row| row.get::<_,i64>(0)).map_err(|e| e.to_string())? as u64;
            Ok(crate::usage::UsageSnapshot { records, longest_chat_ms, unknown_token_requests })
        }).await
    }
    pub async fn performance(
        &self,
        session: String,
        execution: Option<String>,
    ) -> Result<PerformanceSummary, String> {
        self.query(move |connection| {
            let execution = match execution {
                Some(value) => Some(value),
                None => connection.query_row("SELECT id FROM executions WHERE session_id=?1 ORDER BY started_at DESC,id DESC LIMIT 1", [&session], |r| r.get::<_,Option<String>>(0)).optional().map_err(|e| e.to_string())?.flatten()
                    .or(connection.query_row("SELECT execution_id FROM requests WHERE session_id=?1 AND purpose='conversation' ORDER BY started_at DESC,id DESC LIMIT 1", [&session], |r| r.get::<_,Option<String>>(0)).optional().map_err(|e| e.to_string())?.flatten()),
            };
            let get = |execution: Option<&str>| -> Result<(Option<f64>,u64),String> {
                connection.query_row("SELECT SUM(output_tokens)*1000.0/SUM(generation_ms),COUNT(*) FROM requests
                    WHERE session_id=?1 AND purpose='conversation' AND status='success' AND output_tokens>0 AND generation_ms>0 AND tps IS NOT NULL
                    AND (?2 IS NULL OR execution_id=?2)", params![session, execution], |r| Ok((r.get(0)?, r.get::<_,i64>(1)? as u64))).map_err(|e| e.to_string())
            };
            let (session_tps,session_samples) = get(None)?;
            let (last_round_tps,last_round_samples) = if let Some(execution) = &execution { get(Some(execution))? } else { (None,0) };
            let usage = |execution: Option<&str>| -> Result<UsageCounters,String> {
                connection.query_row("SELECT COALESCE(SUM(MAX(COALESCE(input_tokens,0)-COALESCE(cache_read_tokens,0)-COALESCE(cache_write_tokens,0),0)),0),
                    COALESCE(SUM(output_tokens),0),COALESCE(SUM(cache_read_tokens),0),COALESCE(SUM(cache_write_tokens),0)
                    FROM requests WHERE session_id=?1 AND purpose='conversation' AND (?2 IS NULL OR execution_id=?2)", params![session,execution], |row| Ok(UsageCounters {
                        input:row.get::<_,i64>(0)? as u64, output:row.get::<_,i64>(1)? as u64,
                        cache_read:row.get::<_,i64>(2)? as u64,cache_write:row.get::<_,i64>(3)? as u64,
                    })).map_err(|e| e.to_string())
            };
            let session_usage = usage(None)?;
            let last_round_usage = if let Some(execution) = &execution { usage(Some(execution))? } else { UsageCounters::default() };
            let session_rounds = connection.query_row("SELECT COUNT(*) FROM executions WHERE session_id=?1", [&session], |row| row.get::<_,i64>(0)).map_err(|e| e.to_string())? as u64;
            Ok(PerformanceSummary { last_round_tps,session_tps,last_round_samples,session_samples,last_round_usage,session_usage,session_rounds })
        }).await
    }
    pub async fn health(&self, boundaries: Vec<u64>) -> Result<Vec<HealthPoint>, String> {
        if ![8, 31].contains(&boundaries.len())
            || boundaries
                .windows(2)
                .any(|p| p[1] <= p[0] || p[1] - p[0] < 20 * 3600000 || p[1] - p[0] > 28 * 3600000)
        {
            return Err("需要近七天或近三十天递增的本地日期边界".into());
        }
        self.query(move |connection| {
            let mut points = Vec::new();
            let mut statement = connection.prepare("SELECT provider,model,
                SUM(CASE WHEN status='success' AND output_tokens>0 AND generation_ms>0 AND tps IS NOT NULL THEN output_tokens END)*1000.0/
                SUM(CASE WHEN status='success' AND output_tokens>0 AND generation_ms>0 AND tps IS NOT NULL THEN generation_ms END),
                AVG(CASE WHEN status='success' THEN first_token_ms END),SUM(CASE WHEN status='success' AND tps IS NOT NULL AND output_tokens>0 AND generation_ms>0 THEN 1 ELSE 0 END),COUNT(CASE WHEN status='success' THEN first_token_ms END)
                FROM requests WHERE started_at>=?1 AND started_at<?2 GROUP BY provider,model").map_err(|e| e.to_string())?;
            for (day, range) in boundaries.windows(2).enumerate() {
                let rows = statement.query_map(params![integer(range[0])?, integer(range[1])?], |r| Ok(HealthPoint { day, provider:r.get(0)?,model:r.get(1)?,tps:r.get(2)?,first_token_ms:r.get(3)?,tps_samples:r.get::<_,i64>(4)? as u64,first_token_samples:r.get::<_,i64>(5)? as u64, ..Default::default() })).map_err(|e| e.to_string())?;
                for row in rows { points.push(row.map_err(|e| e.to_string())?); }
            }
            Ok(points)
        }).await
    }
    pub async fn health_calls(&self) -> Result<Vec<HealthPoint>, String> {
        self.query(move |connection| {
            let mut statement = connection
                .prepare(
                    "SELECT provider,model,tps,first_token_ms,started_at,id,status FROM (
                        SELECT provider,model,tps,first_token_ms,started_at,id,status FROM requests
                        ORDER BY started_at DESC,id DESC LIMIT 500) ORDER BY started_at,id",
                )
                .map_err(|e| e.to_string())?;
            let rows = statement
                .query_map([], |row| {
                    let tps: Option<f64> = row.get(2)?;
                    let first_token_ms: Option<f64> = row.get(3)?;
                    Ok(HealthPoint {
                        provider: row.get(0)?,
                        model: row.get(1)?,
                        tps,
                        first_token_ms,
                        tps_samples: u64::from(tps.is_some()),
                        first_token_samples: u64::from(first_token_ms.is_some()),
                        timestamp: Some(row.get::<_, i64>(4)? as u64),
                        request_id: row.get(5)?,
                        status: row.get(6)?,
                        ..Default::default()
                    })
                })
                .map_err(|e| e.to_string())?;
            rows.enumerate()
                .map(|(index, row)| {
                    row.map(|mut point| {
                        point.day = index;
                        point
                    })
                    .map_err(|e| e.to_string())
                })
                .collect()
        })
        .await
    }
    pub async fn list(&self, filter: RecordFilter) -> Result<RecordPage, String> {
        if filter.model.as_ref().is_some_and(|id| {
            id.split_once('/').map_or(true, |(provider, model)| {
                provider.is_empty() || model.is_empty()
            })
        }) {
            return Err("无效的模型标识".into());
        }
        if matches!((filter.from_ms,filter.to_ms),(Some(from),Some(to)) if from>=to) {
            return Err("无效的时间范围".into());
        }
        if filter
            .cursor
            .as_ref()
            .is_some_and(|cursor| cursor.id.is_empty())
        {
            return Err("无效的分页游标".into());
        }
        if filter.status.as_ref().is_some_and(|s| {
            !["running", "success", "failed", "cancelled", "interrupted"].contains(&s.as_str())
        }) {
            return Err("无效的请求状态".into());
        }
        self.query(move |connection| {
            let model = filter.model.as_deref().and_then(|id| id.split_once('/'));
            let mut statement = connection.prepare("SELECT id,logical_id,attempt,session_id,execution_id,purpose,provider,model,response_model,
                started_at,ended_at,input_tokens,cache_read_tokens,cache_write_tokens,output_tokens,first_token_ms,generation_ms,tps,status,http_status,error_kind
                FROM requests WHERE (?1 IS NULL OR started_at>=?1) AND (?2 IS NULL OR started_at<?2)
                AND (?3 IS NULL OR provider=?3 AND model=?4) AND (?5 IS NULL OR status=?5)
                AND (?6 IS NULL OR started_at<?6 OR started_at=?6 AND id<?7) ORDER BY started_at DESC,id DESC LIMIT 51").map_err(|e| e.to_string())?;
            let rows = statement.query_map(params![optional_integer(filter.from_ms)?,optional_integer(filter.to_ms)?,model.map(|m|m.0),model.map(|m|m.1),filter.status,optional_integer(filter.cursor.as_ref().map(|c|c.started_at))?,filter.cursor.as_ref().map(|c|c.id.as_str())], request_from_row).map_err(|e|e.to_string())?;
            let mut records = rows.collect::<Result<Vec<_>,_>>().map_err(|e| e.to_string())?;
            let more = records.len()>50;
            records.truncate(50);
            let next_cursor = if more { records.last().map(|r| Cursor { started_at:r.started_at,id:r.id.clone() }) } else { None };
            let models = connection.prepare("SELECT DISTINCT provider||'/'||model FROM requests ORDER BY 1").map_err(|e| e.to_string())?
                .query_map([], |r| r.get(0)).map_err(|e|e.to_string())?.collect::<Result<Vec<String>,_>>().map_err(|e|e.to_string())?;
            Ok(RecordPage { records,next_cursor,models })
        }).await
    }
}
impl RequestObserver for RequestStore {
    fn execution(&self, record: ExecutionRecord) {
        let run_id = self.run_id.clone();
        let error = self.error.clone();
        let notify = self.notify.clone();
        let session = record.session_id.clone();
        let job: Job = Box::new(move |connection| {
            let result = connection.and_then(|connection| {
                connection.execute("INSERT INTO executions (id,run_id,session_id,started_at,ended_at,duration_ms,status)
                    VALUES (?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(id) DO UPDATE SET ended_at=excluded.ended_at,
                    duration_ms=excluded.duration_ms,status=excluded.status WHERE executions.status='running'",
                    params![record.id, run_id, record.session_id, integer(record.started_at)?, optional_integer(record.ended_at)?, optional_integer(record.duration_ms)?, record.status])
                    .map_err(|e| e.to_string())?;
                Ok(())
            });
            *error.lock().unwrap_or_else(|p| p.into_inner()) = result.err();
            notify(session);
        });
        let result = self
            .tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|tx| tx.send(job));
        if !matches!(result, Some(Ok(()))) {
            *self.error.lock().unwrap_or_else(|p| p.into_inner()) =
                Some("请求记录线程已停止".into());
            (self.notify)(String::new());
        }
    }
    fn record(&self, record: RequestRecord) {
        let run_id = self.run_id.clone();
        let error = self.error.clone();
        let notify = self.notify.clone();
        let session = record.context.session_id.clone();
        let job: Job = Box::new(move |connection| {
            let result = connection.and_then(|connection| {
                connection.execute("INSERT INTO requests (id,run_id,logical_id,attempt,session_id,execution_id,purpose,provider,model,
                    started_at,ended_at,status,input_tokens,cache_read_tokens,cache_write_tokens,output_tokens,first_token_ms,generation_ms,tps,http_status,error_kind,response_model)
                    VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22)
                    ON CONFLICT(id) DO UPDATE SET ended_at=excluded.ended_at,status=excluded.status,input_tokens=excluded.input_tokens,cache_read_tokens=excluded.cache_read_tokens,
                    cache_write_tokens=excluded.cache_write_tokens,output_tokens=excluded.output_tokens,first_token_ms=excluded.first_token_ms,generation_ms=excluded.generation_ms,
                    tps=excluded.tps,http_status=excluded.http_status,error_kind=excluded.error_kind,response_model=excluded.response_model
                    WHERE requests.status='running'", params![record.id,run_id,record.logical_id,record.attempt,record.context.session_id,record.context.execution_id,record.context.purpose,record.provider,record.model,integer(record.started_at)?,optional_integer(record.ended_at)?,record.status,optional_integer(record.input_tokens)?,optional_integer(record.cache_read_tokens)?,optional_integer(record.cache_write_tokens)?,optional_integer(record.output_tokens)?,record.first_token_ms,record.generation_ms,record.tps,record.http_status,record.error_kind,record.response_model]).map_err(|e|e.to_string())?;
                Ok(())
            });
            *error.lock().unwrap_or_else(|p| p.into_inner()) = result.err();
            notify(session);
        });
        let result = self
            .tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|tx| tx.send(job));
        if !matches!(result, Some(Ok(()))) {
            *self.error.lock().unwrap_or_else(|p| p.into_inner()) =
                Some("请求记录线程已停止".into());
            (self.notify)(record_context_fallback());
        }
    }
}
fn request_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RequestRecord> {
    let unsigned = |index, value: i64| {
        u64::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(index, value))
    };
    let optional_unsigned = |index| {
        row.get::<_, Option<i64>>(index)?
            .map(|value| unsigned(index, value))
            .transpose()
    };
    Ok(RequestRecord {
        id: row.get(0)?,
        logical_id: row.get(1)?,
        attempt: row.get(2)?,
        context: RequestContext {
            session_id: row.get(3)?,
            execution_id: row.get(4)?,
            purpose: row.get(5)?,
        },
        provider: row.get(6)?,
        model: row.get(7)?,
        response_model: row.get(8)?,
        started_at: unsigned(9, row.get(9)?)?,
        ended_at: optional_unsigned(10)?,
        input_tokens: optional_unsigned(11)?,
        cache_read_tokens: optional_unsigned(12)?,
        cache_write_tokens: optional_unsigned(13)?,
        output_tokens: optional_unsigned(14)?,
        first_token_ms: row.get(15)?,
        generation_ms: row.get(16)?,
        tps: row.get(17)?,
        status: row.get(18)?,
        http_status: row.get(19)?,
        error_kind: row.get(20)?,
    })
}
fn record_context_fallback() -> String {
    String::new()
}
impl Drop for RequestStore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn integer(value: u64) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| "请求数值超出存储范围".into())
}
fn optional_integer(value: Option<u64>) -> Result<Option<i64>, String> {
    value.map(integer).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::request_metrics::RequestContext;
    fn record(
        id: &str,
        started_at: u64,
        execution: &str,
        purpose: &str,
        status: &str,
        tokens: u64,
        duration: f64,
    ) -> RequestRecord {
        RequestRecord {
            id: id.into(),
            logical_id: id.into(),
            attempt: 1,
            context: RequestContext {
                session_id: "session".into(),
                execution_id: Some(execution.into()),
                purpose: purpose.into(),
            },
            provider: "test".into(),
            model: "model".into(),
            response_model: None,
            started_at,
            ended_at: Some(started_at + 1000),
            input_tokens: Some(10),
            cache_read_tokens: Some(5),
            cache_write_tokens: Some(0),
            output_tokens: Some(tokens),
            first_token_ms: Some(500.0),
            generation_ms: Some(duration),
            tps: Some(tokens as f64 / duration * 1000.0),
            status: status.into(),
            http_status: Some(200),
            error_kind: None,
        }
    }
    #[tokio::test]
    async fn weighted_averages_rounds_purposes_and_restart() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("requests.sqlite3");
        let store = RequestStore::open(path.clone(), |_| {});
        store.record(record(
            "one",
            100,
            "round1",
            "conversation",
            "success",
            100,
            9000.0,
        ));
        store.record(record(
            "two",
            200,
            "round2",
            "conversation",
            "success",
            200,
            1000.0,
        ));
        store.record(record(
            "failed",
            300,
            "round2",
            "conversation",
            "failed",
            90000,
            1.0,
        ));
        store.record(record(
            "name", 400, "naming", "naming", "success", 90000, 1.0,
        ));
        let stats = store
            .performance("session".into(), Some("round2".into()))
            .await
            .unwrap();
        assert_eq!(stats.session_tps, Some(30.0));
        assert_eq!(stats.last_round_tps, Some(200.0));
        assert_eq!(stats.session_samples, 2);
        assert_eq!(stats.last_round_samples, 1);
        assert_eq!(
            store
                .performance("session".into(), Some("new".into()))
                .await
                .unwrap()
                .last_round_tps,
            None
        );
        let days = (0..8).map(|day| day * 86400000).collect();
        let health = store.health(days).await.unwrap();
        assert_eq!(health[0].tps_samples, 3);
        assert_eq!(health[0].first_token_samples, 3);
        store.shutdown();
        let reopened = RequestStore::open(path, |_| {});
        assert_eq!(
            reopened
                .performance("session".into(), None)
                .await
                .unwrap()
                .last_round_tps,
            Some(200.0)
        );
        assert_eq!(
            reopened
                .list(RecordFilter::default())
                .await
                .unwrap()
                .records
                .len(),
            4
        );
    }
    #[tokio::test]
    async fn upserts_pagination_filters_and_unknown_samples() {
        let temp = tempfile::tempdir().unwrap();
        let store = RequestStore::open(temp.path().join("requests.sqlite3"), |_| {});
        let mut pending = record(
            "pending",
            100,
            "round",
            "conversation",
            "running",
            100,
            1000.0,
        );
        pending.ended_at = None;
        pending.tps = None;
        pending.generation_ms = None;
        pending.output_tokens = None;
        store.record(pending.clone());
        let mut ended = pending.clone();
        ended.status = "failed".into();
        ended.ended_at = Some(200);
        ended.response_model = Some("model-resolved".into());
        ended.http_status = Some(503);
        ended.error_kind = Some("http".into());
        store.record(ended.clone());
        store.record(pending); // a late duplicate must not resurrect running
        for i in 0..55 {
            store.record(record(
                &format!("{i:03}"),
                200,
                "round",
                "conversation",
                "success",
                100,
                1000.0,
            ));
        }
        let first = store.list(RecordFilter::default()).await.unwrap();
        assert_eq!(first.records.len(), 50);
        let second = store
            .list(RecordFilter {
                cursor: first.next_cursor,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(second.records.len(), 6);
        assert!(
            first
                .records
                .iter()
                .all(|a| second.records.iter().all(|b| a.id != b.id))
        );
        let failed = store
            .list(RecordFilter {
                status: Some("failed".into()),
                model: Some("test/model".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(failed.records.len(), 1);
        assert_eq!(failed.records[0].output_tokens, None);
        assert_eq!(
            serde_json::to_value(&failed.records[0]).unwrap(),
            serde_json::to_value(&ended).unwrap()
        );
        assert!(
            first
                .records
                .iter()
                .all(|record| record.response_model.is_none())
        );
        let columns = store
            .query(|connection| {
                connection
                    .prepare("SELECT name FROM pragma_table_info('requests')")
                    .map_err(|error| error.to_string())?
                    .query_map([], |row| row.get::<_, String>(0))
                    .map_err(|error| error.to_string())?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| error.to_string())
            })
            .await
            .unwrap();
        assert!(!columns.iter().any(|column| column == "payload"));
        assert!(columns.iter().any(|column| column == "response_model"));
        let stats = store.performance("session".into(), None).await.unwrap();
        assert_eq!(stats.session_samples, 55);
        assert!(store.health(vec![0, 1]).await.is_err());
    }
    #[tokio::test]
    async fn concurrent_workers_and_storage_errors_do_not_panic() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("requests.sqlite3");
        let one = RequestStore::open(path.clone(), |_| {});
        one.list(RecordFilter::default()).await.unwrap();
        let two = RequestStore::open(path, |_| {});
        one.record(record(
            "one",
            100,
            "r",
            "conversation",
            "success",
            100,
            1000.0,
        ));
        two.record(record(
            "two",
            200,
            "r",
            "conversation",
            "success",
            100,
            1000.0,
        ));
        two.list(RecordFilter::default()).await.unwrap();
        assert_eq!(
            one.list(RecordFilter::default())
                .await
                .unwrap()
                .records
                .len(),
            2
        );
        let broken = RequestStore::open(temp.path().to_owned(), |_| {});
        assert!(broken.list(RecordFilter::default()).await.is_err());
        assert!(broken.error().is_some());
    }
    #[tokio::test]
    async fn health_ranges_and_calls_keep_all_attempts_and_stable_time_order() {
        let temp = tempfile::tempdir().unwrap();
        let store = RequestStore::open(temp.path().join("requests.sqlite3"), |_| {});
        let day = 86400000;
        store.record(record(
            "old",
            1,
            "r",
            "conversation",
            "success",
            100,
            1000.0,
        ));
        for index in 0..64 {
            let mut request = record(
                &format!("c{index:03}"),
                23 * day + 100,
                "r",
                "conversation",
                "success",
                100,
                1000.0,
            );
            if index == 0 {
                request.status = "failed".into();
                request.tps = None;
                request.first_token_ms = None;
            }
            store.record(request);
        }
        store.record(record(
            "outside",
            30 * day,
            "r",
            "conversation",
            "success",
            100,
            1000.0,
        ));
        let thirty = store
            .health((0..=30).map(|index| index * day).collect())
            .await
            .unwrap();
        assert_eq!(thirty.len(), 2);
        assert_eq!(
            thirty
                .iter()
                .find(|point| point.day == 23)
                .unwrap()
                .tps_samples,
            63
        );
        let seven: Vec<_> = (23..=30).map(|index| index * day).collect();
        let calls = store.health_calls().await.unwrap();
        assert_eq!(calls.len(), 66); // no date limit or 50-row detail-page limit
        assert_eq!(calls[0].request_id.as_deref(), Some("old"));
        assert_eq!(calls[1].request_id.as_deref(), Some("c000"));
        assert_eq!(calls[1].status.as_deref(), Some("failed"));
        assert_eq!(calls[1].tps, None);
        assert_eq!(calls[64].request_id.as_deref(), Some("c063"));
        assert_eq!(calls[64].day, 64);
        assert!(
            calls
                .iter()
                .skip(1)
                .take(64)
                .all(|point| point.timestamp == Some(23 * day + 100))
        );
        assert_eq!(store.health(seven).await.unwrap().len(), 1);
        assert!(
            store
                .health((0..=14).map(|index| index * day).collect())
                .await
                .is_err()
        );
        for index in 0..600 {
            store.record(record(
                &format!("n{index:03}"),
                31 * day + index,
                "r",
                "conversation",
                "success",
                100,
                1000.0,
            ));
        }
        let latest = store.health_calls().await.unwrap();
        assert_eq!(latest.len(), 500);
        assert_eq!(latest[0].request_id.as_deref(), Some("n100"));
        assert_eq!(latest[499].request_id.as_deref(), Some("n599"));
        assert!(
            latest
                .windows(2)
                .all(|pair| pair[0].timestamp <= pair[1].timestamp)
        );
    }
    #[tokio::test]
    async fn sqlite_usage_includes_reported_attempts_and_durable_execution_durations() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("requests.sqlite3");
        let store = RequestStore::open(path.clone(), |_| {});
        store.record(record(
            "one",
            100,
            "r1",
            "conversation",
            "success",
            20,
            1000.0,
        ));
        let mut failed = record("two", 200, "r1", "compaction", "failed", 30, 1000.0);
        failed.input_tokens = None;
        failed.cache_read_tokens = None;
        store.record(failed);
        let mut running = ExecutionRecord {
            id: "r1".into(),
            session_id: "session".into(),
            started_at: 100,
            ended_at: None,
            duration_ms: None,
            status: "running".into(),
        };
        store.execution(running.clone());
        running.status = "success".into();
        running.ended_at = Some(1000);
        running.duration_ms = Some(900);
        store.execution(running.clone());
        running.status = "running".into();
        running.ended_at = None;
        running.duration_ms = None;
        store.execution(running);
        store.execution(ExecutionRecord {
            id: "r2".into(),
            session_id: "session".into(),
            started_at: 2000,
            ended_at: Some(3000),
            duration_ms: Some(1000),
            status: "cancelled".into(),
        });
        store.execution(ExecutionRecord {
            id: "r3".into(),
            session_id: "another".into(),
            started_at: 2000,
            ended_at: Some(3500),
            duration_ms: Some(1500),
            status: "success".into(),
        });
        let usage = store.usage().await.unwrap();
        assert_eq!(usage.records.len(), 2);
        assert_eq!(
            usage
                .records
                .iter()
                .map(|record| record.tokens)
                .sum::<u64>(),
            60
        ); // 10 total input (cache included) + 20 output + 30 reported failure output
        assert_eq!(usage.unknown_token_requests, 1);
        assert_eq!(usage.longest_chat_ms, 1900);
        let stats = store.performance("session".into(), None).await.unwrap();
        assert_eq!(stats.session_rounds, 2);
        assert_eq!(stats.last_round_samples, 0); // latest execution has no request; do not reuse an older round
        assert_eq!(stats.session_usage.input, 5);
        assert_eq!(stats.session_usage.cache_read, 5);
        assert_eq!(stats.session_usage.output, 20);
        store.shutdown();
        let reopened = RequestStore::open(path, |_| {});
        assert_eq!(reopened.usage().await.unwrap().longest_chat_ms, 1900);
        let empty = RequestStore::open(temp.path().join("empty.sqlite3"), |_| {});
        let empty_usage = empty.usage().await.unwrap();
        assert!(empty_usage.records.is_empty());
        assert_eq!(empty_usage.longest_chat_ms, 0);
        empty.shutdown();
        let existing = Connection::open(temp.path().join("empty.sqlite3")).unwrap();
        existing
            .execute_batch("DROP TABLE executions; PRAGMA user_version=1;")
            .unwrap();
        drop(existing);
        let upgraded = RequestStore::open(temp.path().join("empty.sqlite3"), |_| {});
        assert!(upgraded.usage().await.unwrap().records.is_empty());
    }
    #[tokio::test]
    async fn recovery_checks_dead_processes_and_preserves_live_instances() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("requests.sqlite3");
        let store = RequestStore::open(path.clone(), |_| {});
        store.record(record(
            "live",
            100,
            "r",
            "conversation",
            "running",
            0,
            1000.0,
        ));
        store.list(RecordFilter::default()).await.unwrap();
        store.execution(ExecutionRecord {
            id: "running-round".into(),
            session_id: "session".into(),
            started_at: 100,
            ended_at: None,
            duration_ms: None,
            status: "running".into(),
        });
        let second = RequestStore::open(path.clone(), |_| {});
        assert_eq!(
            second.list(RecordFilter::default()).await.unwrap().records[0].status,
            "running"
        );
        second
            .query(|c| {
                c.execute("UPDATE instances SET pid=4294967295", [])
                    .map_err(|e| e.to_string())?;
                Ok(())
            })
            .await
            .unwrap();
        second.shutdown();
        store.shutdown();
        let reopened = RequestStore::open(path, |_| {});
        let records = reopened.list(RecordFilter::default()).await.unwrap();
        assert_eq!(records.records[0].status, "interrupted");
        assert_eq!(
            records.records[0].error_kind.as_deref(),
            Some("process_exit")
        );
        assert!(records.records[0].ended_at.is_some());
        reopened.query(|connection| {
            let (status,ended,duration): (String,Option<i64>,Option<i64>)=connection.query_row("SELECT status,ended_at,duration_ms FROM executions WHERE id='running-round'",[],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).map_err(|e|e.to_string())?;
            assert_eq!(status,"interrupted");assert_eq!(ended,None);assert_eq!(duration,None);
            Ok(())
        }).await.unwrap();
    }
}

// Process creation identity prevents PID reuse from protecting stale records.
#[cfg(windows)]
fn process_identity(pid: u32) -> Result<Option<i64>, ()> {
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, FILETIME, GetLastError},
        System::Threading::{
            GetExitCodeProcess, GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        },
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return if GetLastError() == ERROR_INVALID_PARAMETER {
                Ok(None)
            } else {
                Err(())
            };
        }
        let mut created = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exited = created;
        let mut kernel = created;
        let mut user = created;
        let mut code = 0;
        let times = GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user);
        let running = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        if times == 0 || running == 0 {
            return Err(());
        }
        if code != 259 {
            return Ok(None);
        }
        Ok(Some(
            ((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime)) as i64,
        ))
    }
}
#[cfg(target_os = "linux")]
fn process_identity(pid: u32) -> Result<Option<i64>, ()> {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat
            .rsplit_once(')')
            .and_then(|(_, fields)| fields.split_whitespace().nth(19))
            .and_then(|s| s.parse().ok())
            .map(|v| Some(v))
            .ok_or(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(()),
    }
}
#[cfg(not(any(windows, target_os = "linux")))]
fn process_identity(pid: u32) -> Result<Option<i64>, ()> {
    use std::hash::{Hash, Hasher};
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "lstart="])
        .output()
        .map_err(|_| ())?;
    let text = String::from_utf8_lossy(&output.stdout);
    if !output.stderr.is_empty() {
        return Err(());
    }
    if text.trim().is_empty() {
        return Ok(None);
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.trim().hash(&mut hasher);
    Ok(Some((hasher.finish() & i64::MAX as u64) as i64))
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use step_harness_providers::request_metrics::RequestContext;
    #[tokio::test]
    async fn day_boundaries_and_failed_only_models_preserve_missing_samples() {
        let temp = tempfile::tempdir().unwrap();
        let store = RequestStore::open(temp.path().join("requests.sqlite3"), |_| {});
        let mut boundaries = vec![0];
        for hours in [24, 23, 24, 25, 24, 24, 24] {
            boundaries.push(boundaries.last().unwrap() + hours * 3600000);
        }
        for (id, time, status, model) in [
            ("before", boundaries[1] - 1, "success", "good"),
            ("after", boundaries[1], "success", "good"),
            ("failed", boundaries[3], "failed", "broken"),
            ("outside", boundaries[7], "success", "good"),
        ] {
            store.record(RequestRecord {
                id: id.into(),
                logical_id: id.into(),
                attempt: 1,
                context: RequestContext {
                    session_id: "s".into(),
                    execution_id: Some("r".into()),
                    purpose: "conversation".into(),
                },
                provider: "p".into(),
                model: model.into(),
                response_model: None,
                started_at: time,
                ended_at: Some(time + 1),
                input_tokens: None,
                cache_read_tokens: None,
                cache_write_tokens: None,
                output_tokens: Some(100),
                first_token_ms: Some(200.0),
                generation_ms: Some(1000.0),
                tps: Some(100.0),
                status: status.into(),
                http_status: Some(200),
                error_kind: None,
            });
        }
        let points = store.health(boundaries).await.unwrap();
        assert_eq!(points.len(), 3);
        assert!(points.iter().any(|p| p.day == 0 && p.tps == Some(100.0)));
        assert!(points.iter().any(|p| p.day == 1 && p.tps == Some(100.0)));
        let failed = points.iter().find(|p| p.model == "broken").unwrap();
        assert_eq!(failed.tps, None);
        assert_eq!(failed.first_token_ms, None);
        assert_eq!(failed.tps_samples, 0);
        assert_eq!(failed.first_token_samples, 0);
    }
}
