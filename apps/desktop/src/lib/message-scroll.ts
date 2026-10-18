/** Keep a transcript pinned through streaming, image loads and viewport resizes. */
export function observeMessageScroll(
  viewport: HTMLDivElement,
  content: HTMLDivElement,
  onPinnedChange: (pinned: boolean) => void,
) {
  let pinned = true;
  let lastScrollTop = viewport.scrollTop;

  function setPinned(value: boolean) {
    if (pinned === value) return;
    pinned = value;
    onPinnedChange(value);
  }

  function follow() {
    if (!pinned) return;
    // scrollHeight includes bottom padding, unlike a visible end sentinel.
    viewport.scrollTop = viewport.scrollHeight;
    lastScrollTop = viewport.scrollTop;
  }

  function onScroll() {
    const nearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 100;
    if (nearBottom) setPinned(true);
    // Content growth can produce scroll events too. Only upward movement
    // away from the bottom should suspend following.
    else if (viewport.scrollTop < lastScrollTop) setPinned(false);
    lastScrollTop = viewport.scrollTop;
  }

  const observer = new ResizeObserver(follow);
  observer.observe(content);
  observer.observe(viewport);
  viewport.addEventListener("scroll", onScroll);
  follow();

  return {
    follow,
    jumpToLatest() {
      setPinned(true);
      // An instant jump cannot be mistaken for an upward scroll while a
      // smooth animation is still far from its destination.
      follow();
    },
    dispose() {
      observer.disconnect();
      viewport.removeEventListener("scroll", onScroll);
    },
  };
}
