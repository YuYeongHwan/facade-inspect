"use client";

import { useEffect, useState } from "react";

type State<T> = { key: string | null; data?: T; error?: string };

/** 클라이언트에서만 로드 (api.ts의 상대 URL은 서버 렌더 중엔 쓸 수 없다). key가 바뀌면 다시 로드, null이면 대기. */
export function useData<T>(key: string | null, load: () => Promise<T>) {
  const [state, setState] = useState<State<T>>({ key: null });
  useEffect(() => {
    if (key === null) return;
    let alive = true;
    load().then(
      (data) => alive && setState({ key, data }),
      (e: unknown) => alive && setState({ key, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
    // load는 key로 식별된다 (호출부에서 key에 입력값을 모두 넣는다)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const current = state.key === key;
  return { data: current ? state.data : undefined, error: current ? state.error : undefined, loading: !current };
}
