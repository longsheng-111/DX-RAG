'use client';

import React, { useState, useCallback } from 'react';
import { ConfigProvider, Layout, Typography } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import SessionPanel from '@/components/SessionPanel';
import KnowledgeBaseManager from '@/components/KnowledgeBaseManager';
import FileUpload from '@/components/FileUpload';
import QAPanel from '@/components/QAPanel';
import ExaminerPanel from '@/components/ExaminerPanel';
import FileManager from '@/components/FileManager';
import NewSessionModal from '@/components/NewSessionModal';

const { Content } = Layout;
const { Text } = Typography;

export type MenuKey = 'knowledge-base' | 'upload' | 'qa' | 'files';

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

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [activeMenu, setActiveMenu] = useState<MenuKey>('qa');
  const [selectedCollection, setSelectedCollection] = useState('knowledge_chunks');
  const [activeSessionId, setActiveSessionId] = useState('');
  const [activePersona, setActivePersona] = useState('default');
  const [sessionTotal, setSessionTotal] = useState(0);
  const [refreshSessions, setRefreshSessions] = useState(0);
  const [qaMode, setQaMode] = useState<'qa' | 'examiner'>('qa');
  const [newSessionOpen, setNewSessionOpen] = useState(false);

  const handleCreateSession = useCallback(() => {
    setNewSessionOpen(true);
  }, []);

  const handleSessionCreated = useCallback((session: Session) => {
    setActiveSessionId(session.session_id);
    setActivePersona(session.persona || 'default');
    setSessionTotal(session.total_tokens || 0);
    setSelectedCollection(session.kb_id || selectedCollection);
    const mode = session.mode || 'qa';
    setQaMode(mode);
    setRefreshSessions((n) => n + 1);
    setNewSessionOpen(false);
  }, [selectedCollection]);

  const handleSelectSession = useCallback((session: Session) => {
    setActiveSessionId(session.session_id);
    setActivePersona(session.persona);
    setSessionTotal(session.total_tokens || 0);
    setSelectedCollection(session.kb_id || selectedCollection);
    setQaMode(session.mode || 'qa');
  }, [selectedCollection]);

  const handleSessionUpdate = useCallback((sessionId: string, persona: string, totalTokens: number) => {
    // 只接受当前会话的更新，防止切换会话后迟到的响应覆盖新会话状态
    if (sessionId !== activeSessionId) return;
    setActivePersona(persona);
    setSessionTotal(totalTokens);
    setRefreshSessions((n) => n + 1);
  }, [activeSessionId]);

  const renderContent = () => {
    if (activeMenu === 'qa') {
      return (
        <div style={{ display: 'flex', height: '100%', gap: 0 }}>
          {/* Session Panel */}
          <div style={{
            width: 200,
            flexShrink: 0,
            borderRight: '1.5px solid var(--ink)',
            background: 'var(--bg-panel)',
            height: '100%',
            zIndex: 1,
          }}>
            <SessionPanel
              activeSessionId={activeSessionId}
              onSelectSession={handleSelectSession}
              onCreateSession={handleCreateSession}
              refreshTrigger={refreshSessions}
            />
          </div>
          {/* Chat Area */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-paper)' }}>
            <div style={{
              padding: '10px 16px',
              borderBottom: '1px solid var(--border)',
              background: 'var(--bg-panel)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--ink)',
                  borderBottom: '2px solid var(--brand)',
                  paddingBottom: 2,
                }}
              >
                {qaMode === 'qa' ? '知识问答' : '模拟面试'}
              </span>
              <Text style={{ fontSize: 12, color: 'var(--ink-secondary)' }}>
                {qaMode === 'qa' ? '用户提问，AI 检索回答' : 'AI 出题，用户回答并评分'}
              </Text>
            </div>
            <div style={{ flex: 1, overflowX: 'auto', overflowY: 'auto' }}>
              {activeSessionId ? (
                qaMode === 'qa' ? (
                  <QAPanel
                    sessionId={activeSessionId}
                    persona={activePersona}
                    sessionTotal={sessionTotal}
                    collectionName={selectedCollection}
                    onCollectionChange={setSelectedCollection}
                    onSessionUpdate={handleSessionUpdate}
                  />
                ) : (
                  <ExaminerPanel
                    sessionId={activeSessionId}
                    collectionName={selectedCollection}
                  />
                )
              ) : (
                <div className="empty-state">
                  <div className="empty-state-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 19l7-7 3 3-7 7h-3v-3z" />
                      <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                      <path d="M2 2l7 7" />
                      <path d="M15 5l4 4" />
                    </svg>
                  </div>
                  <span className="empty-state-title">先选一个存档</span>
                  <span className="empty-state-desc">
                    在左侧新建或选择会话，向作业本里的讲解员提问
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }
    switch (activeMenu) {
      case 'knowledge-base':
        return (
          <div className="page-container">
            <KnowledgeBaseManager
              selectedCollection={selectedCollection}
              onSelectCollection={setSelectedCollection}
            />
          </div>
        );
      case 'upload':
        return (
          <div className="page-container page-container--narrow">
            <FileUpload collectionName={selectedCollection} />
          </div>
        );
      case 'files':
        return (
          <div className="page-container">
            <FileManager
              collectionName={selectedCollection}
              onCollectionChange={setSelectedCollection}
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <html lang="zh-CN">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=ZCOOL+KuaiLe&family=Press+Start+2P&family=JetBrains+Mono&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div style={{ position: 'relative', zIndex: 1, minHeight: '100vh' }}>
          <ConfigProvider
            locale={zhCN}
            theme={{
              token: {
                colorPrimary: '#C8392B',
                colorText: '#2B2419',
                colorBgLayout: '#F7EDD8',
                colorBgContainer: '#FFFBF0',
                colorBorder: '#2B2419',
                colorBorderSecondary: 'rgba(43,36,25,0.15)',
                borderRadius: 3,
                fontSize: 14,
                boxShadow: '3px 3px 0 #2B2419',
                boxShadowSecondary: '3px 3px 0 rgba(43,36,25,0.25)',
              },
              components: {
                Button: { borderRadius: 3, primaryShadow: 'none' },
                Modal: { borderRadiusLG: 3 },
                Tag: { borderRadiusSM: 3 },
              },
            }}
          >
            <Layout style={{ minHeight: '100vh', background: 'var(--bg-paper)' }}>
              <Sidebar
                activeMenu={activeMenu}
                onMenuChange={setActiveMenu}
              />
              <Layout style={{ background: 'transparent' }}>
                <Content style={{ height: '100vh', overflow: 'hidden' }}>
                  {renderContent()}
                </Content>
              </Layout>
            </Layout>
            <NewSessionModal
              open={newSessionOpen}
              onCancel={() => setNewSessionOpen(false)}
              onCreated={handleSessionCreated}
              defaultCollection={selectedCollection}
            />
          </ConfigProvider>
        </div>
      </body>
    </html>
  );
}
