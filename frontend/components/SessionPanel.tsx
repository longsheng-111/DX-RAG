'use client';

import React, { useEffect, useState, useCallback } from 'react';
import {
  Button, List, Tag, Typography, Popconfirm, Spin, Tooltip,
} from 'antd';
import {
  PlusOutlined, DeleteOutlined, MessageOutlined,
  DownloadOutlined, ClockCircleOutlined,
} from '@ant-design/icons';
import axios from 'axios';

const { Text } = Typography;

interface Session {
  session_id: string;
  title: string;
  persona: string;
  kb_id: string;
  mode?: 'qa' | 'examiner';
  exam_state?: any;
  total_tokens: number;
  updated_at: string;
  message_count: number;
}

interface Persona { id: string; name: string; }

interface Props {
  activeSessionId: string;
  onSelectSession: (session: Session) => void;
  onCreateSession: () => void;
  refreshTrigger?: number;
}

export default function SessionPanel({
  activeSessionId, onSelectSession, onCreateSession, refreshTrigger = 0,
}: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [personas, setPersonas] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  const fetchPersonas = useCallback(async () => {
    try {
      const res = await axios.get('/api/personas');
      const map: Record<string, string> = {};
      (res.data.personas || []).forEach((p: Persona) => { map[p.id] = p.name; });
      setPersonas(map);
    } catch { /* ignore */ }
  }, []);

  const fetchSessions = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get('/api/sessions');
      setSessions(res.data.sessions || []);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchPersonas(); fetchSessions(); }, [fetchPersonas, fetchSessions, refreshTrigger]);

  const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    try {
      await axios.delete(`/api/sessions/${sessionId}`);
      const remaining = sessions.filter((s) => s.session_id !== sessionId);
      if (sessionId === activeSessionId) {
        if (remaining.length > 0) {
          onSelectSession(remaining[0]);
        } else {
          onCreateSession();
        }
      }
      fetchSessions();
    } catch { /* ignore */ }
  };

  const handleExport = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    try {
      const res = await axios.get(`/api/sessions/${sessionId}/export`, { responseType: 'blob' });
      const blob = new Blob([res.data], { type: 'text/markdown' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url; link.download = `session_${sessionId}.md`;
      document.body.appendChild(link); link.click();
      document.body.removeChild(link); window.URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  };

  const formatTime = (iso: string) => {
    try {
      const d = new Date(iso);
      // 非法时间字符串会产生 Invalid Date，回退显示原始字符串
      if (isNaN(d.getTime())) return iso || '--';
      return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    } catch { return iso; }
  };

  const stickerRotation = (index: number) => index % 2 === 0 ? -2 : 2;

  return (
    <div className="dx-session-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{
        padding: '14px 14px 10px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <MessageOutlined style={{ color: 'var(--brand)', fontSize: 18 }} />
          <Text strong style={{ fontSize: 15, color: 'var(--ink)', letterSpacing: '-0.2px' }}>
            会话
          </Text>
        </div>
        <Tooltip title="新建会话">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={onCreateSession}
            style={{
              borderRadius: '3px !important',
              height: 32,
              width: 32,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 0,
              background: 'var(--brand) !important',
              border: '1.5px solid var(--ink) !important',
              boxShadow: 'none !important',
            }}
          />
        </Tooltip>
      </div>

      {/* List */}
      <div style={{ flex: 1, overflow: 'auto', padding: '0 10px 12px' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>
        ) : sessions.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            paddingTop: 48,
            gap: 12,
          }}>
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ color: 'var(--ink-faint)' }}>
              <path d="M12 52L24 40L20 36L8 48V52H12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M24 40L48 16L44 12L20 36L24 40Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M48 16L52 20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M44 12H52V20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M8 52H56" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.4" />
            </svg>
            <Text style={{ color: 'var(--ink-secondary)', fontSize: 14 }}>
              还没有存档，新建一个开始闯关
            </Text>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              size="small"
              onClick={onCreateSession}
              style={{
                borderRadius: '3px !important',
                background: 'var(--brand) !important',
                border: '1.5px solid var(--ink) !important',
                boxShadow: 'none !important',
              }}
            >
              开始闯关
            </Button>
          </div>
        ) : (
          <List
            dataSource={sessions}
            renderItem={(session, index) => {
              const isActive = activeSessionId === session.session_id;
              const slotNum = String(index + 1).padStart(2, '0');
              return (
                <div
                  onClick={() => onSelectSession(session)}
                  className={`dx-session-card ${isActive ? 'dx-session-card-active' : ''}`}
                  style={{
                    padding: '10px 12px',
                    marginBottom: 8,
                    cursor: 'pointer',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 200ms cubic-bezier(0.25, 0.8, 0.25, 1)',
                  }}
                >
                  <div
                    className="dx-session-actions"
                    style={{
                      position: 'absolute',
                      top: 6,
                      right: 6,
                      display: 'flex',
                      gap: 2,
                      opacity: isActive ? 1 : 0,
                      transition: 'opacity 200ms cubic-bezier(0.25, 0.8, 0.25, 1)',
                      zIndex: 2,
                    }}
                  >
                    <Tooltip title="导出">
                      <Button type="text" size="small"
                        style={{
                          width: 22, height: 22, padding: 0, borderRadius: 3,
                          border: 'none',
                          background: 'transparent',
                        }}
                        icon={<DownloadOutlined style={{ fontSize: 11, color: 'var(--ink-secondary)' }} />}
                        onClick={(e) => handleExport(e, session.session_id)} />
                    </Tooltip>
                    <Popconfirm title="删除该会话？" onConfirm={(e) => handleDelete(e as any, session.session_id)}
                      okText="删除" cancelText="取消">
                      <Button type="text" size="small"
                        style={{
                          width: 22, height: 22, padding: 0, borderRadius: 3,
                          border: 'none',
                          background: 'transparent',
                        }}
                        icon={<DeleteOutlined style={{ fontSize: 11, color: 'var(--brand)' }} />}
                        onClick={(e) => e.stopPropagation()} />
                    </Popconfirm>
                  </div>
                  <div style={{ marginBottom: 6, paddingRight: 28 }}>
                    <Text strong style={{ fontSize: 13, lineHeight: '1.4', color: 'var(--ink)' }} ellipsis>
                      <span style={{ fontFamily: '"ZCOOL KuaiLe", cursive' }}>存档</span>
                      <span style={{ fontFamily: '"Press Start 2P", monospace', marginLeft: 6, marginRight: 8 }}>{slotNum}</span>
                      {session.title || '新会话'}
                    </Text>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 4px', marginBottom: 6 }}>
                    <Tag
                      className="dx-session-sticker"
                      style={{
                        fontSize: 10,
                        lineHeight: '18px',
                        border: '1.5px solid var(--ink) !important',
                        borderRadius: '3px !important',
                        color: 'var(--ink)',
                        background: 'var(--bg-paper) !important',
                        margin: 0,
                        fontWeight: 500,
                        transform: `rotate(${stickerRotation(index)}deg)`,
                        transformOrigin: 'center center',
                        transition: 'transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 200ms cubic-bezier(0.34, 1.56, 0.64, 1)',
                      }}
                    >
                      {personas[session.persona] || session.persona}
                    </Tag>
                    {session.mode === 'examiner' && (
                      <Tag
                        className="dx-session-sticker"
                        style={{
                          fontSize: 10,
                          lineHeight: '18px',
                          border: '1.5px solid var(--ink) !important',
                          borderRadius: '3px !important',
                          margin: 0,
                          color: 'var(--ink)',
                          background: 'var(--bg-paper) !important',
                          fontWeight: 500,
                          transform: `rotate(${-stickerRotation(index)}deg)`,
                          transformOrigin: 'center center',
                          transition: 'transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 200ms cubic-bezier(0.34, 1.56, 0.64, 1)',
                        }}
                      >
                        模拟面试
                      </Tag>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <ClockCircleOutlined style={{ fontSize: 10, color: 'var(--ink-faint)', flexShrink: 0 }} />
                    <Text style={{ fontSize: 10, color: 'var(--ink-faint)', whiteSpace: 'nowrap' }}>
                      {formatTime(session.updated_at)} · {session.message_count} 条消息
                    </Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Text style={{
                      fontSize: 11,
                      color: 'var(--ink-faint)',
                      fontWeight: 500,
                      whiteSpace: 'nowrap',
                      fontVariantNumeric: 'tabular-nums',
                      fontFamily: '"Press Start 2P", monospace',
                    }}>
                      {session.total_tokens.toLocaleString()} tokens
                    </Text>
                  </div>
                </div>
              );
            }}
          />
        )}
      </div>

      <style>{`
        .dx-session-card {
          background: var(--bg-paper);
          border: 1.5px solid var(--ink);
          border-radius: 3px;
          box-shadow: none;
        }
        .dx-session-card-active {
          background: var(--brand-soft) !important;
          border-color: var(--ink) !important;
          box-shadow: none !important;
        }
        .dx-session-card:hover {
          transform: translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
        }
        .dx-session-card-active:hover {
          transform: none;
          box-shadow: none !important;
        }
        .dx-session-card:hover .dx-session-actions,
        .dx-session-card-active .dx-session-actions {
          opacity: 1 !important;
        }
        .dx-session-sticker:hover {
          transform: rotate(0deg) translate(-1px, -1px) !important;
          box-shadow: 3px 3px 0 var(--ink) !important;
        }
        .dx-session-sticker:active {
          transform: rotate(0deg) translate(0, 0) !important;
          box-shadow: none !important;
        }
        .dx-session-panel .ant-btn-primary:hover {
          background: var(--brand-hover) !important;
          box-shadow: 3px 3px 0 var(--ink) !important;
          transform: translate(-1px, -1px);
        }
        .dx-session-panel .ant-btn-primary:active {
          background: var(--brand-hover) !important;
          box-shadow: none !important;
          transform: translate(0, 0);
        }
        @media (prefers-reduced-motion: reduce) {
          .dx-session-card,
          .dx-session-sticker,
          .dx-session-panel .ant-btn-primary {
            transition-duration: 100ms !important;
          }
          .dx-session-card:hover,
          .dx-session-card-active:hover,
          .dx-session-sticker:hover {
            transform: none !important;
            box-shadow: none !important;
          }
        }
      `}</style>
    </div>
  );
}
