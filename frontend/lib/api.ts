import axios from 'axios';
import { message } from 'antd';

/**
 * 从 FastAPI 错误响应中提取可读文本。
 * 注意 422 校验错误的 detail 是数组（[{ loc, msg, ... }]），不能直接弹对象。
 */
export function formatApiError(err: any, fallback = '请求失败'): string {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d: any) => {
        if (typeof d === 'string') return d;
        if (d?.msg) return Array.isArray(d.loc) ? `${d.loc.join('.')}：${d.msg}` : d.msg;
        return JSON.stringify(d);
      })
      .join('；');
  }
  if (err?.code === 'ECONNABORTED') return '请求超时，请稍后重试';
  return err?.message || fallback;
}

declare module 'axios' {
  export interface AxiosRequestConfig {
    /** 置为 true 时不弹统一错误提示，由调用方自行处理错误 */
    skipErrorToast?: boolean;
  }
}

// LLM 请求耗时长，统一给 90s 超时；错误提示在响应拦截器里收口
const api = axios.create({ timeout: 90_000 });

api.interceptors.response.use(
  (res) => res,
  (err) => {
    // 主动取消的请求（如切换会话时 abort）不提示
    if (!axios.isCancel(err) && !err.config?.skipErrorToast) {
      message.error(formatApiError(err));
    }
    return Promise.reject(err);
  },
);

export default api;
