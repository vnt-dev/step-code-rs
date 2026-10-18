import { parseDrafts, type Draft } from "./drafts";

// IndexedDB avoids the small localStorage quota when drafts contain image data.
async function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("step-code-drafts", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("drafts");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function loadImageDrafts(): Promise<Record<string, Draft> | null> {
  const db = await database();
  try {
    return await new Promise((resolve, reject) => {
      const request = db.transaction("drafts").objectStore("drafts").get("projects");
      request.onsuccess = () => resolve(request.result == null ? null : parseDrafts(JSON.stringify(request.result)));
      request.onerror = () => reject(request.error);
    });
  } finally { db.close(); }
}

export async function saveImageDrafts(drafts: Record<string, Draft>): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("drafts", "readwrite");
      transaction.objectStore("drafts").put(drafts, "projects");
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally { db.close(); }
}
