import type { ImageBlock } from "../types";

export interface ImageAttachment extends ImageBlock { id: string; name: string; pasted?: boolean }
export const imageTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
export const maxImageBytes = 10 * 1024 * 1024;
export const maxImages = 8;
export const imageSource = (image: ImageBlock) => `data:${image.mimeType};base64,${image.data}`;

const imageExtensions: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
};

/** Download name for an image that carries no file name of its own. */
export function imageFileName(image: ImageBlock, index: number): string {
  return `图片-${index + 1}.${imageExtensions[image.mimeType] ?? "png"}`;
}

export function isImageAttachment(value: unknown): value is ImageAttachment {
  if (!value || typeof value !== "object") return false;
  const image = value as ImageAttachment;
  return image.type === "image" && typeof image.id === "string" && typeof image.name === "string" &&
    imageTypes.includes(image.mimeType) && typeof image.data === "string" && image.data.length > 0 &&
    image.data.length <= Math.ceil(maxImageBytes / 3) * 4 && /^[A-Za-z0-9+/]+={0,2}$/.test(image.data);
}

export async function readImage(file: File, pasted = false): Promise<ImageAttachment> {
  if (!imageTypes.includes(file.type)) throw new Error("请选择 PNG、JPEG、GIF 或 WebP 图片。");
  if (!file.size || file.size > maxImageBytes) throw new Error("每张图片需大于 0 且不超过 10 MB。");
  const data = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error(`无法读取图片：${file.name}`));
    reader.readAsDataURL(file);
  });
  return { type: "image", id: crypto.randomUUID(), name: file.name || "粘贴的图片", mimeType: file.type, data, pasted };
}
