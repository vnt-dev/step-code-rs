//! Full-screen viewer for the images attached to a message: enlarge, browse and download.

import { useEffect, useRef } from "react";
import type { ImageBlock } from "../types";
import { imageFileName, imageSource } from "../lib/images";
import Icon from "./Icon";

interface ImageLightboxProps {
  images: ImageBlock[];
  index: number;
  onIndexChange: (index: number) => void;
  onClose: () => void;
}

export default function ImageLightbox({ images, index, onIndexChange, onClose }: ImageLightboxProps) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  const step = (delta: number) => onIndexChange((index + delta + images.length) % images.length);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    return () => previous?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
      else if (event.key === "ArrowLeft") step(-1);
      else if (event.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const download = () => {
    const link = document.createElement("a");
    link.href = imageSource(images[index]);
    link.download = imageFileName(images[index], index);
    document.body.append(link);
    link.click();
    link.remove();
  };

  return <div className="lightbox-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) close.current(); }}>
    <div ref={ref} className="lightbox-panel" role="dialog" aria-modal="true" aria-label="图片预览" tabIndex={-1} onKeyDown={event => {
      if (event.key !== "Tab") return;
      const items = Array.from(ref.current?.querySelectorAll<HTMLElement>("button:not(:disabled)") ?? []);
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }}>
      <img className="lightbox-image" src={imageSource(images[index])} alt={`附加图片 ${index + 1}`} />
      <div className="lightbox-toolbar">
        {images.length > 1 && <button type="button" aria-label="上一张图片" onClick={() => step(-1)}><Icon name="chevronLeft" /></button>}
        {images.length > 1 && <span className="lightbox-counter">{index + 1} / {images.length}</span>}
        <button type="button" onClick={download}><Icon name="download" />下载图片</button>
        {images.length > 1 && <button type="button" aria-label="下一张图片" onClick={() => step(1)}><Icon name="chevronRight" /></button>}
        <button type="button" className="lightbox-close" aria-label="关闭预览" onClick={onClose}>×</button>
      </div>
    </div>
  </div>;
}
