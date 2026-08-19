'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Input, Button, Select, Typography, Space, Tooltip,
  Avatar, Modal, Tag, message, Alert, Radio,
} from 'antd';
import {
  QuestionCircleOutlined,
  SwapOutlined,
  FlagOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import {
  Send,
  Trash2,
  FileText,
  Database,
  User,
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  HelpCircle,
  Check,
} from 'lucide-react';
import TextareaAutosize from 'react-textarea-autosize';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import api, { formatApiError } from '@/lib/api';

const { Text } = Typography;
const { TextArea } = Input;

// ============================================================
//  Types
// ============================================================

interface TokenCost {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  session_total: number;
}

interface Source {
  id: string;
  content: string;
  similarity: number;
  file_name: string;
}

interface ReasoningStep {
  step: number;
  action: string;
  query: string;
  hits: number;
  top_score: number;
  confidence: 'high' | 'medium' | 'low';
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  reasoning_steps?: ReasoningStep[];
  queries?: string[];
  token_cost?: TokenCost;
}

interface Collection {
  name: string;
  chunk_count: number;
}

interface Persona {
  id: string;
  name: string;
  description: string;
  default_kb: string;
}

interface Props {
  sessionId: string;
  persona: string;
  sessionTotal: number;
  collectionName: string;
  onCollectionChange: (name: string) => void;
  onSessionUpdate: (sessionId: string, persona: string, totalTokens: number) => void;
}

interface Cluster {
  key: string;
  label: string;
  sources: Source[];
}

interface PopupState {
  visible: boolean;
  pinned: boolean;
  sourceId: string | null;
  citationNum: number | null;
  msgIdx: number | null;
  anchorEl: HTMLSpanElement | null;
}

// ============================================================
//  Constants
// ============================================================

const SESSION_BUDGET = 50000;

const FLAG_OPTIONS = [
  { value: '引用不相关', label: '引用不相关' },
  { value: '与原文不符', label: '与原文不符' },
  { value: '其他', label: '其他' },
];

// ============================================================
//  Helpers
// ============================================================

function clusterSources(sources: Source[]): Cluster[] {
  // 按来源文件分组，避免硬编码主题关键词与知识库内容不匹配
  const groups = new Map<string, Source[]>();
  sources.forEach((s) => {
    if (!groups.has(s.file_name)) groups.set(s.file_name, []);
    groups.get(s.file_name)!.push(s);
  });

  return Array.from(groups.entries()).map(([fileName, group]) => ({
    key: fileName,
    label: fileName,
    sources: group,
  }));
}

function getSourceClusterIndex(sourceId: string, clusters: Cluster[]): number {
  return clusters.findIndex((c) => c.sources.some((s) => s.id === sourceId));
}

function getSourceById(sourceId: string, sources: Source[]): Source | undefined {
  return sources.find((s) => s.id === sourceId);
}

function sourceIndexToId(index: number, sources: Source[]): string | undefined {
  return sources[index - 1]?.id;
}

function getGlobalSourceIndex(sourceId: string, sources: Source[]): number {
  return sources.findIndex((s) => s.id === sourceId) + 1;
}

function getCiteColorVar(index: number): string {
  return `var(--cite-${((Math.max(0, index - 1) % 6) + 1)})`;
}

function scoreColorClass(score: number): string {
  if (score >= 0.85) return 'var(--brand)';
  if (score >= 0.7) return 'var(--ink)';
  return 'var(--ink-faint)';
}

function colorWithOpacity(cssVar: string, opacity: number): string {
  const map: Record<string, string> = {
    'var(--cite-1)': '47,155,232',
    'var(--cite-2)': '139,92,246',
    'var(--cite-3)': '124,181,24',
    'var(--cite-4)': '229,165,10',
    'var(--cite-5)': '232,93,158',
    'var(--cite-6)': '24,169,153',
  };
  const rgb = map[cssVar];
  return rgb ? `rgba(${rgb},${opacity})` : cssVar;
}

function getSummaryLines(content: string, maxLines: number): string[] {
  return content.split('\n').filter((l) => l.trim() !== '').slice(0, maxLines);
}

// ============================================================
//  Sub-components
// ============================================================

function ReasoningTimeline({ steps }: { steps: ReasoningStep[] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="qa-reasoning">
      <div className="qa-reasoning-summary" onClick={() => setExpanded(!expanded)}>
        <Space size={8}>
          <Bot size={14} style={{ color: 'var(--brand)' }} />
          <span>推理路径</span>
          <span className="qa-step-count">{steps.length} 步</span>
        </Space>
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </div>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.25, 0.8, 0.25, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <div className="qa-reasoning-steps">
              {steps.map((step) => (
                <div key={step.step} className="qa-step">
                  <div className="qa-step-num">{step.step}</div>
                  <div className="qa-step-body">
                    <div className="qa-step-action">
                      {step.action === 'knowledge_search' ? '检索知识库' : '改写查询'}
                    </div>
                    <div className="qa-step-detail">
                      查询：{step.query}
                      {step.action === 'knowledge_search' && (
                        <span style={{ marginLeft: 12 }}>
                          命中 {step.hits} 条 · 最高分 {(step.top_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <div className={`qa-step-confidence qa-confidence-${step.confidence}`}>
                    {step.confidence === 'high' ? '高置信' : step.confidence === 'medium' ? '中置信' : '低置信'}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CitationPopup({
  source,
  globalIdx,
  visible,
  pinned,
  onMouseEnter,
  onMouseLeave,
  onClose,
  onReplace,
  onInaccurate,
}: {
  source?: Source;
  globalIdx: number;
  visible: boolean;
  pinned: boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onClose: () => void;
  onReplace: () => void;
  onInaccurate: () => void;
}) {
  if (!visible || !source) return null;

  const summary = getSummaryLines(source.content, pinned ? Number.MAX_SAFE_INTEGER : 6);

  return (
    <div
      className={`qa-citation-popup ${pinned ? 'qa-citation-popup-pinned' : 'qa-citation-popup-preview'}`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{ borderTop: `4px solid ${getCiteColorVar(globalIdx)}` }}
    >
      <div className="qa-citation-popup-header">
        <Space size={8}>
          <span
            className="qa-citation-popup-badge"
            style={{ background: getCiteColorVar(globalIdx) }}
          >
            {globalIdx}
          </span>
          <span className="qa-citation-popup-file" title={source.file_name}>{source.file_name}</span>
          <span className="qa-citation-popup-score" style={{ color: scoreColorClass(source.similarity) }}>
            {(source.similarity * 100).toFixed(0)}%
          </span>
        </Space>
        {pinned && (
          <button className="qa-citation-popup-close" onClick={onClose} title="关闭">
            <CloseOutlined />
          </button>
        )}
      </div>
      <div className="qa-citation-popup-body" style={pinned ? { maxHeight: 320, overflowY: 'auto' } : undefined}>
        {summary.map((line, idx) => (
          <div key={idx} className="qa-citation-popup-line">{line}</div>
        ))}
      </div>
      {pinned && (
        <div className="qa-citation-popup-actions">
          <Button size="small" icon={<SwapOutlined style={{ fontSize: 12 }} />} onClick={onReplace}>
            替换此引用
          </Button>
          <Button size="small" icon={<FlagOutlined style={{ fontSize: 12 }} />} onClick={onInaccurate}>
            标记不准确
          </Button>
        </div>
      )}
    </div>
  );
}

interface CitationTextProps {
  value: string;
  sources: Source[];
  msgIdx: number;
  activeId?: string;
  pinnedId?: string;
  hoveredCiteIndex: number | null;
  pulsingCitationIdx: number | null;
  onDropReplace?: (fromSourceId: string, toCitationNum: number) => void;
  citationOverrides?: Record<number, string>;
  onRegisterCitation: (msgIdx: number, globalIdx: number, el: HTMLSpanElement) => void;
  onUnregisterCitation: (msgIdx: number, globalIdx: number, el: HTMLSpanElement) => void;
  onCitationEnter: (sourceId: string, globalIdx: number, el: HTMLSpanElement, citationNum: number, msgIdx: number) => void;
  onCitationLeave: () => void;
  onCitationClick: (sourceId: string, globalIdx: number, el: HTMLSpanElement, citationNum: number, msgIdx: number) => void;
}

function CitationText({
  value,
  sources,
  msgIdx,
  activeId,
  pinnedId,
  hoveredCiteIndex,
  pulsingCitationIdx,
  onDropReplace,
  citationOverrides,
  onRegisterCitation,
  onUnregisterCitation,
  onCitationEnter,
  onCitationLeave,
  onCitationClick,
}: CitationTextProps) {
  const parts = value.split(/(\[\^\d+\^\])/g);

  return (
    <>
      {parts.map((part, idx) => {
        const match = part.match(/\[\^(\d+)\^\]/);
        if (!match) return <span key={idx}>{part}</span>;
        const num = parseInt(match[1], 10);
        let sourceId = sourceIndexToId(num, sources);
        let displayNum = num;

        // 如果用户通过"替换引用"覆盖了当前 citation，则显示目标来源编号
        const overrideTargetId = citationOverrides?.[num];
        if (overrideTargetId && sources) {
          const targetIdx = sources.findIndex((s) => s.id === overrideTargetId);
          if (targetIdx >= 0) {
            displayNum = targetIdx + 1;
            sourceId = overrideTargetId;
          }
        }

        const globalIdx = sourceId ? getGlobalSourceIndex(sourceId, sources) : 0;
        const isActive = sourceId === activeId;
        const isPinned = sourceId === pinnedId;
        const isHovered = hoveredCiteIndex !== null && hoveredCiteIndex === globalIdx;
        const isPulsing = pulsingCitationIdx !== null && pulsingCitationIdx === globalIdx;
        const className = `qa-citation ${isActive ? 'qa-citation-active' : ''} ${isPinned ? 'qa-citation-pinned' : ''} ${isHovered ? 'qa-citation-hovered' : ''} ${isPulsing ? 'qa-citation-pulse' : ''}`;
        const rotation = globalIdx % 2 === 1 ? -2 : 2;

        return (
          <CitationSpan
            key={idx}
            className={className}
            style={{ background: getCiteColorVar(globalIdx), '--citation-rotation': `${rotation}deg` } as React.CSSProperties}
            sourceId={sourceId}
            globalIdx={globalIdx}
            displayNum={displayNum}
            citationNum={num}
            msgIdx={msgIdx}
            title={`来源 ${displayNum}${sourceId ? ': ' + (getSourceById(sourceId, sources)?.file_name || '') : ''}`}
            onRegister={onRegisterCitation}
            onUnregister={onUnregisterCitation}
            onMouseEnter={onCitationEnter}
            onMouseLeave={onCitationLeave}
            onClick={onCitationClick}
            onDropReplace={onDropReplace}
          />
        );
      })}
    </>
  );
}

// react-markdown v9 的 components 只覆盖元素节点（text 节点不经过自定义组件），
// 因此在元素 children 层面把字符串子节点交给 CitationText 注入引用角标。
// 注意：组件类型必须在模块级保持稳定——若在渲染中动态创建组件类型，
// 每次渲染都会卸载重建整个 markdown 子树，使已捕获的锚点 DOM 脱离文档（浮层定位退化为 0,0）。
// 所以引用相关 props 通过 Context 传递，组件类型保持模块级常量。
const CitationPropsContext = React.createContext<Omit<CitationTextProps, 'value'> | null>(null);

function CitationChildren({ children }: { children: React.ReactNode }) {
  const ctx = React.useContext(CitationPropsContext);
  if (!ctx) return <>{children}</>;
  return (
    <>
      {React.Children.map(children, (child, i) =>
        typeof child === 'string' ? (
          <CitationText key={i} value={child} {...ctx} />
        ) : (
          child
        )
      )}
    </>
  );
}

const citedElement = (Tag: 'p' | 'li' | 'h1' | 'h2' | 'h3' | 'h4' | 'td' | 'th' | 'strong' | 'em') =>
  function CitedElement(props: { children?: React.ReactNode }) {
    return (
      <Tag>
        <CitationChildren>{props.children}</CitationChildren>
      </Tag>
    );
  };

const citationMarkdownComponents = {
  p: citedElement('p'),
  li: citedElement('li'),
  h1: citedElement('h1'),
  h2: citedElement('h2'),
  h3: citedElement('h3'),
  h4: citedElement('h4'),
  td: citedElement('td'),
  th: citedElement('th'),
  strong: citedElement('strong'),
  em: citedElement('em'),
};

function CitationSpan({
  className,
  style,
  sourceId,
  globalIdx,
  displayNum,
  citationNum,
  msgIdx,
  title,
  onRegister,
  onUnregister,
  onMouseEnter,
  onMouseLeave,
  onClick,
  onDropReplace,
}: {
  className: string;
  style: React.CSSProperties;
  sourceId?: string;
  globalIdx: number;
  displayNum: number;
  citationNum: number;
  msgIdx: number;
  title: string;
  onRegister: (msgIdx: number, globalIdx: number, el: HTMLSpanElement) => void;
  onUnregister: (msgIdx: number, globalIdx: number, el: HTMLSpanElement) => void;
  onMouseEnter: (sourceId: string, globalIdx: number, el: HTMLSpanElement, citationNum: number, msgIdx: number) => void;
  onMouseLeave: () => void;
  onClick: (sourceId: string, globalIdx: number, el: HTMLSpanElement, citationNum: number, msgIdx: number) => void;
  onDropReplace?: (fromSourceId: string, toCitationNum: number) => void;
}) {
  const spanRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = spanRef.current;
    if (el) onRegister(msgIdx, globalIdx, el);
    return () => {
      if (el) onUnregister(msgIdx, globalIdx, el);
    };
  }, [msgIdx, globalIdx, onRegister, onUnregister]);

  if (!sourceId) return <span className={className} style={style}>{displayNum}</span>;

  return (
    <span
      ref={spanRef}
      className={className}
      style={style}
      onMouseEnter={() => {
        if (spanRef.current) onMouseEnter(sourceId, globalIdx, spanRef.current, citationNum, msgIdx);
      }}
      onMouseLeave={onMouseLeave}
      onClick={(e) => {
        e.stopPropagation();
        if (spanRef.current) onClick(sourceId, globalIdx, spanRef.current, citationNum, msgIdx);
      }}
      draggable={false}
      onDragOver={(e) => {
        if (onDropReplace) e.preventDefault();
      }}
      onDrop={(e) => {
        if (!onDropReplace) return;
        e.preventDefault();
        const fromId = e.dataTransfer.getData('text/source-id');
        if (fromId) {
          onDropReplace(fromId, citationNum);
        }
      }}
      title={title}
    >
      {displayNum}
    </span>
  );
}



function SourcePanel({
  sources,
  clusters,
  activeId,
  pinnedId,
  hoveredCiteIndex,
  pulsingSourceId,
  filterFile,
  onFilterChange,
  onActivate,
  onPin,
  onRef,
  onSourceBadgeClick,
  panelCollapsed,
  onToggle,
}: {
  sources: Source[];
  clusters: Cluster[];
  activeId?: string;
  pinnedId?: string;
  hoveredCiteIndex: number | null;
  pulsingSourceId: string | null;
  filterFile: string;
  onFilterChange: (file: string) => void;
  onActivate: (id: string | undefined, globalIdx?: number) => void;
  onPin: (id: string | undefined) => void;
  onRef: (id: string, el: HTMLDivElement | null) => void;
  onSourceBadgeClick: (globalIdx: number) => void;
  panelCollapsed: boolean;
  onToggle: () => void;
}) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const visibleClusters = filterFile === 'all' || filterFile === 'current_kb'
    ? clusters
    : filterFile === 'selected_files'
      ? (pinnedId ? clusters.filter((c) => c.sources.some((s) => s.id === pinnedId)) : [])
      : clusters.filter((c) => c.label === filterFile);
  const visibleCount = visibleClusters.reduce((sum, c) => sum + c.sources.length, 0);

  const filterOptions = [
    { value: 'all', label: '全部' },
    { value: 'current_kb', label: '仅当前知识库' },
    { value: 'selected_files', label: '仅选中文件' },
  ];

  if (panelCollapsed) {
    return (
      <div className="qa-retrieval-panel collapsed">
        <button
          className="qa-panel-toggle"
          onClick={onToggle}
          title="展开检索结果"
        >
          <ChevronRight size={18} />
        </button>
      </div>
    );
  }

  return (
    <div className="qa-retrieval-panel">
      <div className="qa-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span className="qa-panel-title">找到的线索</span>
          <Tooltip title="按来源范围过滤下方片段">
            <QuestionCircleOutlined style={{ color: 'var(--ink-faint)', fontSize: 12 }} />
          </Tooltip>
          <Select
            value={filterFile}
            onChange={(val) => onFilterChange(val)}
            options={filterOptions}
            size="small"
            className="qa-source-filter"
            classNames={{ popup: { root: 'qa-source-filter-dropdown' } }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <span className="qa-panel-count">
            共 {visibleCount} 条
          </span>
          <button
            className="qa-panel-toggle"
            onClick={onToggle}
            title="收起检索结果"
          >
            <ChevronLeft size={12} />
          </button>
        </div>
      </div>
      <div className="qa-scroll">
        {visibleClusters.length === 0 && (
          <Text style={{ color: 'var(--ink-faint)', fontSize: 12, padding: 12, lineHeight: 1.6 }}>
            没有匹配当前筛选条件的检索片段
          </Text>
        )}
        {visibleClusters.map((cluster) => {
          const isCollapsed = collapsed[cluster.key];
          return (
            <div key={cluster.key} className="qa-cluster">
              <div
                className="qa-cluster-header"
                onClick={() => setCollapsed((prev) => ({ ...prev, [cluster.key]: !isCollapsed }))}
              >
                <div className="qa-cluster-title">
                  <FileText size={12} style={{ color: 'var(--ink-faint)', flexShrink: 0 }} />
                  <span title={cluster.label}>{cluster.label}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                  <span className="qa-cluster-count">{cluster.sources.length}</span>
                  {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                </div>
              </div>
              <AnimatePresence>
                {!isCollapsed && (
                  <motion.div
                    className="qa-cluster-body"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: [0.25, 0.8, 0.25, 1] }}
                    style={{ overflow: 'hidden' }}
                  >
                    {cluster.sources.map((src) => {
                      const globalIdx = sources.findIndex((s) => s.id === src.id) + 1;
                      const isActive = activeId === src.id;
                      const isPinned = pinnedId === src.id;
                      const isHovered = hoveredCiteIndex === globalIdx;
                      return (
                        <div
                          key={src.id}
                          ref={(el) => onRef(src.id, el)}
                          className={`qa-source ${isActive || isHovered ? 'qa-source-active' : ''} ${isPinned ? 'qa-source-pinned' : ''} ${pulsingSourceId === src.id ? 'qa-source-pulse' : ''}`}
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData('text/source-id', src.id);
                            e.dataTransfer.effectAllowed = 'move';
                          }}
                          onMouseEnter={() => onActivate(src.id, globalIdx)}
                          onMouseLeave={() => onActivate(undefined)}
                          onClick={() => {
                            const willPin = !isPinned;
                            onPin(willPin ? src.id : undefined);
                            if (willPin) onSourceBadgeClick(globalIdx);
                          }}
                          title={isPinned ? '点击取消固定' : '点击固定引用并跳转到答案中的对应引用'}
                          style={{
                            ...(isActive || isHovered ? { background: colorWithOpacity(getCiteColorVar(globalIdx), 0.08) } : {}),
                            ...(pulsingSourceId === src.id ? { '--pulse-color': colorWithOpacity(getCiteColorVar(globalIdx), 0.08) } as React.CSSProperties : {}),
                          }}
                        >
                          <div className="qa-source-meta">
                            <span
                              className="qa-source-id"
                              style={{ background: getCiteColorVar(globalIdx) }}
                              onClick={(e) => {
                                e.stopPropagation();
                                onSourceBadgeClick(globalIdx);
                              }}
                              title="跳转到答案中的引用"
                            >
                              {globalIdx}
                            </span>
                            <span className="qa-source-file" title={src.file_name}>{src.file_name}</span>
                            <span className="qa-source-score" style={{ color: scoreColorClass(src.similarity) }}>
                              {(src.similarity * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="qa-source-text">{src.content}</div>
                        </div>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
//  Main component
// ============================================================

export default function QAPanel({
  sessionId, persona: propsPersona, sessionTotal: propsSessionTotal,
  collectionName, onCollectionChange, onSessionUpdate,
}: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [lastCost, setLastCost] = useState<TokenCost | null>(null);
  const [activeCitationId, setActiveCitationId] = useState<string | undefined>();
  const [pinnedCitationId, setPinnedCitationId] = useState<string | undefined>();
  const [hoveredCiteIndex, setHoveredCiteIndex] = useState<number | null>(null);
  const [pulsingSourceId, setPulsingSourceId] = useState<string | null>(null);
  const [pulsingCitationIdx, setPulsingCitationIdx] = useState<number | null>(null);
  const [citationOverrides, setCitationOverrides] = useState<Record<number, Record<number, string>>>({});
  const [feedbackMsgIdx, setFeedbackMsgIdx] = useState<number | null>(null);
  const [feedbackType, setFeedbackType] = useState<'replace' | 'inaccurate' | null>(null);
  const [feedbackNote, setFeedbackNote] = useState('');
  const [replaceCitationNum, setReplaceCitationNum] = useState<number | null>(null);
  const [replaceTargetId, setReplaceTargetId] = useState('');
  const [retrievalCollapsed, setRetrievalCollapsed] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [flagProblemType, setFlagProblemType] = useState<string | null>(null);
  const [flagSubmitting, setFlagSubmitting] = useState(false);
  const [flagSubmitted, setFlagSubmitted] = useState(false);
  const [feedbackStamps, setFeedbackStamps] = useState<Record<number, boolean>>({});
  const [replaceSubmitting, setReplaceSubmitting] = useState(false);
  const [popup, setPopup] = useState<PopupState>({
    visible: false, pinned: false, sourceId: null, citationNum: null, msgIdx: null, anchorEl: null,
  });
  const sourceRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const citationRefs = useRef<Map<string, HTMLSpanElement>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const popupEnterTimer = useRef<NodeJS.Timeout | null>(null);
  const popupLeaveTimer = useRef<NodeJS.Timeout | null>(null);
  // 进行中的问答请求，切换会话时 abort，避免旧回答串入新会话
  const chatAbortRef = useRef<AbortController | null>(null);
  // 镜像当前 sessionId，供异步回调比对请求发起时的会话
  const sessionIdRef = useRef(sessionId);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  const fetchCollections = useCallback(async () => {
    try {
      const res = await axios.get('/api/collections');
      setCollections(res.data.collections || []);
    } catch { /* ignore */ }
  }, []);

  const fetchPersonas = useCallback(async () => {
    try {
      const res = await axios.get('/api/personas');
      setPersonas(res.data.personas || []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    setActiveCitationId(undefined);
    setPinnedCitationId(undefined);
    setHoveredCiteIndex(null);
    setPopup({ visible: false, pinned: false, sourceId: null, citationNum: null, msgIdx: null, anchorEl: null });
    // 会话切换（或组件卸载）时取消进行中的问答请求
    return () => {
      chatAbortRef.current?.abort();
    };
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    const loadSession = async () => {
      try {
        const res = await axios.get(`/api/sessions/${sessionId}`);
        if (res.data) {
          setMessages((res.data.messages || []).map((m: any) => ({
            role: m.role,
            content: m.content,
            sources: m.sources || undefined,
            reasoning_steps: m.reasoning_steps || undefined,
            queries: m.queries || undefined,
            token_cost: m.token_cost,
          })));
        }
      } catch { /* ignore */ }
    };
    loadSession();
  }, [sessionId]);

  useEffect(() => { fetchCollections(); fetchPersonas(); }, [fetchCollections, fetchPersonas]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!activeCitationId) return;
    const el = sourceRefs.current.get(activeCitationId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeCitationId]);

  // 固定某条引用后，给答案区对应角标一个脉冲提示
  useEffect(() => {
    if (!pinnedCitationId) return;
    const lastAssistant = messages[messages.length - 1];
    if (lastAssistant?.role !== 'assistant' || !lastAssistant.sources) return;
    const idx = getGlobalSourceIndex(pinnedCitationId, lastAssistant.sources);
    if (idx <= 0) return;
    setPulsingCitationIdx(idx);
    const t = setTimeout(() => setPulsingCitationIdx(null), 600);
    return () => clearTimeout(t);
  }, [pinnedCitationId, messages]);

  // Close pinned popup on Esc or click outside
  useEffect(() => {
    if (!popup.pinned) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closePopup();
    };
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;
      const insidePopup = popupRef.current?.contains(target);
      const insideAnchor = popup.anchorEl?.contains(target);
      if (!insidePopup && !insideAnchor) closePopup();
    };
    window.addEventListener('keydown', handleKey);
    document.addEventListener('mousedown', handleClick);
    return () => {
      window.removeEventListener('keydown', handleKey);
      document.removeEventListener('mousedown', handleClick);
    };
  }, [popup.pinned, popup.anchorEl]);

  const clearPopupTimers = useCallback(() => {
    if (popupEnterTimer.current) {
      clearTimeout(popupEnterTimer.current);
      popupEnterTimer.current = null;
    }
    if (popupLeaveTimer.current) {
      clearTimeout(popupLeaveTimer.current);
      popupLeaveTimer.current = null;
    }
  }, []);

  const closePopup = useCallback(() => {
    clearPopupTimers();
    setPopup({ visible: false, pinned: false, sourceId: null, citationNum: null, msgIdx: null, anchorEl: null });
  }, [clearPopupTimers]);

  const handleCitationEnter = useCallback((sourceId: string, globalIdx: number, anchorEl: HTMLSpanElement, citationNum: number, msgIdx: number) => {
    setHoveredCiteIndex(globalIdx);
    setActiveCitationId(sourceId);
    clearPopupTimers();
    popupEnterTimer.current = setTimeout(() => {
      setPopup((p) => p.pinned ? p : { visible: true, pinned: false, sourceId, citationNum, msgIdx, anchorEl });
    }, 200);
  }, [clearPopupTimers]);

  const handleCitationLeave = useCallback(() => {
    setHoveredCiteIndex(null);
    setActiveCitationId((prev) => (pinnedCitationId ? pinnedCitationId : undefined));
    clearPopupTimers();
    popupLeaveTimer.current = setTimeout(() => {
      setPopup((p) => (p.pinned ? p : { ...p, visible: false }));
    }, 150);
  }, [clearPopupTimers, pinnedCitationId]);

  const handleCitationClick = useCallback((sourceId: string, globalIdx: number, anchorEl: HTMLSpanElement, citationNum: number, msgIdx: number) => {
    const willPin = pinnedCitationId !== sourceId;
    setPinnedCitationId(willPin ? sourceId : undefined);
    setActiveCitationId(willPin ? sourceId : undefined);
    clearPopupTimers();
    setPopup({ visible: true, pinned: willPin, sourceId, citationNum, msgIdx, anchorEl });
    setPulsingSourceId(sourceId);
    const el = sourceRefs.current.get(sourceId);
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setTimeout(() => setPulsingSourceId(null), 800);
  }, [clearPopupTimers, pinnedCitationId]);

  const handlePopupEnter = useCallback(() => {
    clearPopupTimers();
  }, [clearPopupTimers]);

  const handlePopupLeave = useCallback(() => {
    clearPopupTimers();
    popupLeaveTimer.current = setTimeout(() => {
      setPopup((p) => (p.pinned ? p : { ...p, visible: false }));
    }, 150);
  }, [clearPopupTimers]);

  const registerCitation = useCallback((msgIdx: number, globalIdx: number, el: HTMLSpanElement) => {
    const key = `${msgIdx}-${globalIdx}`;
    if (!citationRefs.current.has(key)) {
      citationRefs.current.set(key, el);
    }
  }, []);

  const unregisterCitation = useCallback((msgIdx: number, globalIdx: number, el: HTMLSpanElement) => {
    const key = `${msgIdx}-${globalIdx}`;
    if (citationRefs.current.get(key) === el) {
      citationRefs.current.delete(key);
    }
  }, []);

  const handleSourceBadgeClick = useCallback((globalIdx: number) => {
    const entries = Array.from(citationRefs.current.entries())
      .filter(([k]) => k.endsWith(`-${globalIdx}`))
      .sort(([a], [b]) => {
        const ma = parseInt(a.split('-')[0], 10);
        const mb = parseInt(b.split('-')[0], 10);
        return mb - ma;
      });
    entries[0]?.[1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, []);

  const handleSend = async () => {
    const question = inputValue.trim();
    if (!question || loading || !sessionId) return;

    const userMsg: Message = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setLoading(true);
    setLastCost(null);
    setActiveCitationId(undefined);
    setCitationOverrides({});
    setHoveredCiteIndex(null);
    setSourceFilter('all');
    closePopup();

    const requestSessionId = sessionId;
    const controller = new AbortController();
    chatAbortRef.current?.abort();
    chatAbortRef.current = controller;

    try {
      const res = await api.post('/api/chat', {
        session_id: requestSessionId, question, top_k: 5,
        persona: propsPersona, collection_name: collectionName,
      }, { signal: controller.signal, skipErrorToast: true });
      // 响应到达时会话已切换，丢弃旧会话的结果
      if (requestSessionId !== sessionIdRef.current) return;
      const cost: TokenCost = res.data.token_cost;
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources,
        reasoning_steps: res.data.reasoning_steps,
        queries: res.data.queries,
        token_cost: cost,
      }]);
      setLastCost(cost);
      onSessionUpdate(requestSessionId, res.data.persona, cost.session_total);
    } catch (err: any) {
      // 主动取消或会话已切换时静默丢弃，错误提示留在聊天气泡里
      if (axios.isCancel(err) || requestSessionId !== sessionIdRef.current) return;
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: `请求失败：${formatApiError(err)}`,
      }]);
    } finally {
      // 只由最新一次请求复位 loading，避免旧请求误清新请求的 loading
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null;
        setLoading(false);
      }
    }
  };

  const handleClear = async () => {
    if (!sessionId) return;
    try {
      await api.post(`/api/sessions/${sessionId}/clear`);
      setMessages([]);
      setFeedbackStamps({});
    } catch {
      // 拦截器已统一弹出错误提示，本地消息保持不动
    }
  };

  const handleDropReplace = useCallback((fromSourceId: string, toCitationNum: number) => {
    setFeedbackMsgIdx(messages.length - 1);
    setReplaceCitationNum(toCitationNum);
    setReplaceTargetId('');
    setFeedbackType('replace');
  }, [messages.length]);

  const openReplaceModal = useCallback((msgIdx: number, citationNum?: number) => {
    setFeedbackMsgIdx(msgIdx);
    setFeedbackType('replace');
    setReplaceCitationNum(citationNum ?? null);
    setReplaceTargetId('');
  }, []);

  const openFlagModal = useCallback((msgIdx: number, citationNum?: number) => {
    setFeedbackMsgIdx(msgIdx);
    setFeedbackType('inaccurate');
    setReplaceCitationNum(citationNum ?? null);
    setFlagProblemType(null);
    setFeedbackNote('');
    setFlagSubmitted(false);
  }, []);

  const confirmReplace = async () => {
    const effectiveCitationNum = replaceCitationNum ?? 1;
    if (feedbackMsgIdx === null || !replaceTargetId) return;
    const msg = messages[feedbackMsgIdx];
    const originalSource = msg.sources?.[effectiveCitationNum - 1];
    const targetSource = msg.sources?.find((s) => s.id === replaceTargetId);
    if (!originalSource || !targetSource) return;

    setReplaceSubmitting(true);
    setCitationOverrides((prev) => ({
      ...prev,
      [feedbackMsgIdx]: { ...(prev[feedbackMsgIdx] || {}), [effectiveCitationNum]: replaceTargetId },
    }));

    try {
      await axios.post('/api/feedback', {
        session_id: sessionId,
        original_chunk_id: originalSource.id,
        target_chunk_id: targetSource.id,
        type: 'replace',
        original_content: originalSource.content,
        target_content: targetSource.content,
        note: '',
      });
      message.success({ content: '引用已替换，后续回答将参考这条线索', duration: 2 });
    } catch (err: any) {
      message.error(err.response?.data?.detail || '保存反馈失败');
    } finally {
      setReplaceSubmitting(false);
    }

    setFeedbackType(null);
    setFeedbackNote('');
    setReplaceTargetId('');
    setReplaceCitationNum(null);
    setFeedbackMsgIdx(null);
  };

  const submitInaccurate = async () => {
    if (!flagProblemType || feedbackMsgIdx === null) return;
    setFlagSubmitting(true);
    const msg = messages[feedbackMsgIdx];
    const effectiveCitationNum = replaceCitationNum ?? 1;
    const originalSource = msg.sources?.[effectiveCitationNum - 1];
    const note = `[${flagProblemType}] ${feedbackNote.trim()}`.trim();

    try {
      await axios.post('/api/feedback', {
        session_id: sessionId,
        original_chunk_id: originalSource?.id || '',
        target_chunk_id: '',
        type: 'inaccurate',
        original_content: originalSource?.content || '',
        target_content: '',
        note,
      });
      message.success('感谢反馈');
      setFlagSubmitted(true);
      setFeedbackStamps((prev) => ({ ...prev, [feedbackMsgIdx]: true }));
    } catch (err: any) {
      message.error(err.response?.data?.detail || '保存反馈失败');
    } finally {
      setFlagSubmitting(false);
    }
  };

  const budgetRatio = propsSessionTotal / SESSION_BUDGET;
  // 用量低 → 绿，中 → 黄，高 → 红
  const budgetColor = budgetRatio < 0.5 ? 'var(--cite-3)' : budgetRatio < 0.8 ? 'var(--cite-4)' : 'var(--brand)';
  const tokenSegments = 20;
  const usedSegments = Math.min(Math.ceil(budgetRatio * tokenSegments), tokenSegments);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); handleSend(); }
  };

  const collectionOptions = collections.map((c) => ({
    value: c.name, label: `${c.name} (${c.chunk_count})`,
  }));
  if (!collectionOptions.find((o) => o.value === collectionName)) {
    collectionOptions.unshift({ value: collectionName, label: collectionName });
  }

  const lastAssistantIdx = messages.length - 1;

  const popupGlobalIdx = popup.sourceId && popup.msgIdx !== null
    ? getGlobalSourceIndex(popup.sourceId, messages[popup.msgIdx]?.sources || [])
    : 0;
  const popupSource = popup.msgIdx !== null && popup.sourceId
    ? getSourceById(popup.sourceId, messages[popup.msgIdx]?.sources || [])
    : undefined;

  const popupStyle = (() => {
    if (!popup.anchorEl || !rootRef.current) return { display: 'none' };
    const rootRect = rootRef.current.getBoundingClientRect();
    const anchorRect = popup.anchorEl.getBoundingClientRect();
    const popupHeight = popup.pinned ? 360 : 180;
    const popupWidth = 320;
    const anchorCenterX = anchorRect.left - rootRect.left + anchorRect.width / 2;
    const anchorTopY = anchorRect.top - rootRect.top;
    const spaceAbove = anchorTopY;
    const spaceBelow = rootRect.height - anchorTopY - anchorRect.height;
    const showAbove = spaceAbove >= popupHeight + 16 || spaceAbove > spaceBelow;
    let left = anchorCenterX;
    if (left + popupWidth / 2 > rootRect.width - 8) {
      left = rootRect.width - popupWidth / 2 - 8;
    }
    if (left - popupWidth / 2 < 8) {
      left = popupWidth / 2 + 8;
    }
    return {
      position: 'absolute' as const,
      left,
      top: anchorTopY + (showAbove ? 0 : anchorRect.height),
      transform: showAbove ? 'translate(-50%, -100%)' : 'translate(-50%, 0)',
      marginTop: showAbove ? -8 : 8,
      zIndex: 100,
    };
  })();

  const composer = (
    <div className="qa-composer">
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <TextareaAutosize
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="写下你的问题，按 Ctrl + Enter 出发"
          minRows={1}
          maxRows={8}
          style={{
            flex: 1,
            fontSize: 14,
            lineHeight: 1.5,
            resize: 'none',
            padding: '8px 0',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <Tooltip title="发送 (Ctrl+Enter)">
          <Button
            type="primary"
            className="qa-send-btn"
            icon={<Send size={18} />}
            onClick={handleSend}
            loading={loading}
            disabled={!inputValue.trim()}
            style={{
              height: 40,
              width: 40,
              borderRadius: 'var(--radius)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 0,
            }}
          />
        </Tooltip>
      </div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: 6,
      }}>
        <Text style={{ fontSize: 11, color: 'var(--ink-secondary)' }}>
          AI 生成内容仅供参考，关键决策请核对原文
        </Text>
        <Text style={{ fontSize: 11, color: 'var(--ink-secondary)' }}>
          Ctrl + Enter 发送
        </Text>
      </div>
    </div>
  );

  const personaName = personas.find((p) => p.id === propsPersona)?.name || propsPersona;

  return (
    <div ref={rootRef} className="qa-panel-root" style={{
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minHeight: 0,
      padding: '0 12px',
      position: 'relative',
    }}>
      <style>{`
        /* Toolbar */
        .qa-toolbar {
          background: var(--bg-panel);
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
        }
        .qa-persona-sticker {
          display: inline-block;
          max-width: 120px;
          padding: 2px 10px;
          background: var(--bg-panel);
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          font-family: var(--font-display);
          font-size: 12px;
          font-weight: 600;
          color: var(--ink);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          line-height: 1.5;
          transform: rotate(-2deg);
          transition: transform 150ms var(--ease-pop), box-shadow 150ms var(--ease-pop);
          cursor: default;
        }
        .qa-persona-sticker:nth-of-type(even) { transform: rotate(2deg); }
        .qa-persona-sticker:hover {
          transform: rotate(0deg) translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
        }
        .qa-kb-select {
          max-width: 200px;
        }
        .qa-kb-select .ant-select-selector {
          border-radius: var(--radius) !important;
        }
        .qa-kb-select .ant-select-selection-item {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .qa-session-tokens {
          font-variant-numeric: tabular-nums;
        }
        .qa-token-count {
          font-family: var(--font-pixel);
          font-size: 10px;
          line-height: 1;
        }
        .qa-token-bar {
          display: flex;
          gap: 2px;
          align-items: center;
          flex-shrink: 0;
        }
        .qa-token-segment {
          width: 4px;
          height: 10px;
          background: var(--brand);
          border: 1px solid var(--ink);
        }
        .qa-token-segment.empty {
          background: var(--bg-sunken);
        }
        .qa-session-tokens-compact {
          display: none;
          align-items: center;
          gap: 4px;
          font-size: 11px;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
          color: var(--brand);
        }
        .qa-token-round-only {
          display: none;
          font-size: 11px;
          color: var(--ink-secondary);
        }
        @media (max-width: 1279px) {
          .qa-token-round { display: none !important; }
          .qa-token-bar { display: none !important; }
          .qa-token-round-only { display: inline !important; }
          .qa-token-denom { display: none; }
        }
        @media (max-width: 1023px) {
          .qa-token-round-only { display: none !important; }
          .qa-session-tokens { display: none !important; }
          .qa-session-tokens-compact { display: inline-flex !important; }
        }

        /* Composer */
        .qa-composer {
          margin: 0 12px 12px;
          padding: 10px 12px;
          background:
            linear-gradient(rgba(43,36,25,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(43,36,25,0.04) 1px, transparent 1px),
            var(--bg-panel);
          background-size: 20px 20px, 20px 20px, 100% 100%;
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          transition: border-color 150ms var(--ease-base);
        }
        .qa-composer:focus-within {
          border-color: var(--brand);
        }
        .qa-send-btn.ant-btn-primary {
          background: var(--brand) !important;
          border: 1.5px solid var(--ink) !important;
          border-radius: var(--radius) !important;
          box-shadow: 2px 2px 0 var(--ink) !important;
          color: #fff !important;
          transition: transform 100ms var(--ease-base), box-shadow 100ms var(--ease-base), background 150ms var(--ease-base) !important;
        }
        .qa-send-btn.ant-btn-primary svg {
          color: #fff !important;
        }
        .qa-send-btn.ant-btn-primary:hover:not(:disabled) {
          transform: translate(-1px, -1px) !important;
          box-shadow: 3px 3px 0 var(--ink) !important;
          background: var(--brand-hover) !important;
        }
        .qa-send-btn.ant-btn-primary:active:not(:disabled) {
          transform: translate(2px, 2px) !important;
          box-shadow: none !important;
        }
        .qa-send-btn.ant-btn-primary:disabled {
          background: var(--ink-faint) !important;
          border-color: var(--ink-faint) !important;
          color: var(--bg-panel) !important;
          box-shadow: none !important;
        }

        /* Empty state */
        .qa-empty-state {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 12px;
          color: var(--ink-secondary);
          font-size: 14px;
        }
        .qa-empty-pencil {
          color: var(--ink);
        }

        /* Loading skeleton */
        .qa-skeleton {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 0;
        }
        .qa-skeleton-block {
          background: var(--bg-sunken);
          border-radius: 2px;
          height: 10px;
        }

        /* Error alert */
        .qa-error-alert {
          border: 1.5px solid var(--ink) !important;
          background: var(--bg-panel) !important;
          min-height: 36px;
          padding: 6px 12px !important;
        }
        .qa-error-alert .ant-alert-message {
          color: var(--brand) !important;
          font-size: 13px;
        }

        /* Retrieval panel */
        .qa-retrieval-panel {
          background: var(--bg-panel);
          border-right: 1px solid rgba(43,36,25,0.12);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          width: 240px;
          min-width: 240px;
          transition: width 200ms var(--ease-base), min-width 200ms var(--ease-base);
        }
        .qa-retrieval-panel.collapsed {
          width: 44px;
          min-width: 44px;
        }
        .qa-panel-header {
          height: 40px;
          padding: 0 10px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-bottom: 1px solid rgba(43,36,25,0.12);
          background: var(--bg-panel);
          flex-shrink: 0;
        }
        .qa-panel-title {
          font-size: 13px;
          font-weight: 600;
          color: var(--ink);
          white-space: nowrap;
        }
        .qa-panel-count {
          font-size: 11px;
          color: var(--ink-faint);
          font-weight: 400;
          font-variant-numeric: tabular-nums;
        }
        .qa-panel-toggle {
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-panel);
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          cursor: pointer;
          color: var(--ink-secondary);
          transition: transform 150ms var(--ease-base), box-shadow 150ms var(--ease-base), color 150ms var(--ease-base);
          padding: 0;
        }
        .qa-panel-toggle:hover {
          transform: translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
          color: var(--brand);
        }
        .qa-panel-toggle:active {
          transform: translate(0, 0);
          box-shadow: none;
        }
        .qa-source-filter {
          min-width: 110px;
        }
        .qa-source-filter .ant-select-selector {
          border-radius: var(--radius) !important;
          border-color: rgba(43,36,25,0.25) !important;
          background: var(--bg-panel) !important;
          font-size: 12px !important;
        }
        .qa-source-filter .ant-select-selection-item {
          color: var(--ink-secondary) !important;
        }
        .qa-source-filter-dropdown .ant-select-dropdown {
          border: 1.5px solid var(--ink) !important;
          border-radius: var(--radius) !important;
          box-shadow: 3px 3px 0 var(--ink) !important;
        }
        .qa-scroll {
          flex: 1;
          overflow-y: auto;
          padding: 12px;
        }
        .qa-cluster {
          margin-bottom: 12px;
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          background: var(--bg-panel);
          overflow: hidden;
        }
        .qa-cluster-header {
          padding: 8px 10px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
          background: var(--bg-sunken);
          border-bottom: 1px solid rgba(43,36,25,0.12);
        }
        .qa-cluster-title {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
          font-weight: 600;
          color: var(--ink);
          min-width: 0;
        }
        .qa-cluster-title span {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .qa-cluster-count {
          font-size: 11px;
          color: var(--ink-faint);
          font-variant-numeric: tabular-nums;
          background: var(--bg-panel);
          padding: 1px 6px;
          border: 1px solid rgba(43,36,25,0.12);
          border-radius: var(--radius);
        }
        .qa-cluster-body {
          padding: 12px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .qa-source {
          position: relative;
          padding: 10px 12px;
          border-radius: var(--radius);
          border: 1.5px solid var(--ink);
          background: var(--bg-panel);
          cursor: pointer;
          transition: transform 150ms var(--ease-base), box-shadow 150ms var(--ease-base), background 150ms var(--ease-base);
        }
        .qa-source:hover {
          transform: translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
        }
        .qa-source:active {
          transform: translate(0, 0);
          box-shadow: none;
        }
        .qa-source-active {
          border-color: var(--ink) !important;
        }
        .qa-source-pinned {
          border-color: var(--brand) !important;
          box-shadow: 4px 4px 0 var(--ink) !important;
        }
        .qa-source-pinned::after {
          content: '';
          position: absolute;
          top: -5px;
          right: -5px;
          width: 10px;
          height: 10px;
          background: var(--brand);
          border: 1.5px solid var(--ink);
          border-radius: 50%;
        }
        .qa-source-pulse {
          animation: qa-source-pulse-anim 800ms ease-in-out;
        }
        @keyframes qa-source-pulse-anim {
          0% { background: transparent; }
          50% { background: var(--pulse-color, var(--brand-soft)); }
          100% { background: transparent; }
        }
        .qa-source-meta {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 6px;
        }
        .qa-source-id {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 20px;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          font-size: 10px;
          font-weight: 700;
          color: #fff;
          border: 1.5px solid var(--ink);
          flex-shrink: 0;
          cursor: pointer;
          transition: transform 150ms var(--ease-base), box-shadow 150ms var(--ease-base);
        }
        .qa-source-id:hover {
          transform: translate(-1px, -1px);
          box-shadow: 2px 2px 0 var(--ink);
        }
        .qa-source-id:active {
          transform: translate(0, 0);
          box-shadow: none;
        }
        .qa-source-file {
          font-size: 11px;
          color: var(--ink-secondary);
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          flex: 1;
          min-width: 0;
        }
        .qa-source-score {
          font-family: var(--font-pixel);
          font-size: 10px;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
          flex-shrink: 0;
        }
        .qa-source-text {
          font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
          font-size: 12px;
          line-height: 1.55;
          color: var(--ink-secondary);
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        /* Answer area */
        .qa-answer-area {
          flex: 1;
          min-width: 0;
          min-height: 0;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          background: var(--bg-paper);
        }
        .qa-answer-card {
          background: var(--bg-panel);
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          margin-bottom: 16px;
          padding: 3px;
          overflow: hidden;
          position: relative;
        }
        .qa-answer-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 12px;
          height: 12px;
          border-top: 3px solid var(--brand);
          border-left: 3px solid var(--brand);
          pointer-events: none;
          z-index: 2;
        }
        .qa-answer-card-inner {
          border: 1px solid var(--ink);
          border-radius: var(--radius);
          overflow: hidden;
        }
        .qa-answer-toolbar {
          height: 40px;
          padding: 0 12px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-bottom: 1px solid rgba(43,36,25,0.12);
          background: var(--bg-sunken);
        }
        .qa-answer-toolbar-label {
          font-family: var(--font-display);
          font-size: 13px;
          font-weight: 600;
          color: var(--ink);
        }
        .qa-answer-body {
          padding: 16px;
          font-size: 14px;
          line-height: 32px;
          color: var(--ink);
          position: relative;
          background: repeating-linear-gradient(
            transparent,
            transparent 31px,
            rgba(43,36,25,0.08) 31px,
            rgba(43,36,25,0.08) 32px
          );
        }
        .qa-user-bubble {
          max-width: 80%;
          padding: 10px 14px;
          background: var(--brand-soft);
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          color: var(--ink);
          font-size: 14px;
          line-height: 1.6;
        }

        /* Citations */
        .qa-citation {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          font-size: 10px;
          font-weight: 700;
          color: #fff;
          vertical-align: super;
          margin: 0 1px;
          cursor: pointer;
          border: 1.5px solid var(--ink);
          transform: rotate(var(--citation-rotation, 0deg));
          transition: transform 200ms var(--ease-pop), box-shadow 150ms var(--ease-base);
          user-select: none;
        }
        .qa-citation:hover {
          transform: rotate(0deg) translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
        }
        .qa-citation:active {
          transform: rotate(0deg) translate(0, 0);
          box-shadow: none;
        }
        .qa-citation-active,
        .qa-citation-pinned {
          box-shadow: 0 0 0 2px var(--bg-panel), 0 0 0 4px var(--brand);
          outline: none;
        }
        .qa-citation-hovered {
          box-shadow: 0 0 0 2px var(--bg-panel), 0 0 0 4px var(--ink);
          outline: none;
        }
        .qa-citation-pulse {
          animation: qa-citation-pulse-anim 600ms ease-in-out;
        }
        @keyframes qa-citation-pulse-anim {
          0%, 100% { box-shadow: 0 0 0 2px var(--bg-panel), 0 0 0 4px var(--brand); }
          50% { box-shadow: 0 0 0 4px var(--bg-panel), 0 0 0 8px var(--brand); }
        }

        /* Citation popup */
        .qa-citation-popup {
          /* 保持 in-flow：外壳 popupStyle 负责定位，本类若 absolute 会使外壳塌缩 0×0、translate(-50%) 退化（v1.9.6） */
          width: 320px;
          background: var(--bg-panel);
          border: 1.5px solid var(--ink);
          border-radius: 3px;
          box-shadow: 4px 4px 0 var(--ink);
          pointer-events: auto;
          overflow: hidden;
        }
        .qa-citation-popup-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 12px;
          border-bottom: 1px solid rgba(43,36,25,0.15);
          background: var(--bg-sunken);
        }
        .qa-citation-popup-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 18px;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          font-size: 10px;
          font-weight: 700;
          color: #fff;
          border: 1.5px solid var(--ink);
        }
        .qa-citation-popup-file {
          font-size: 12px;
          font-weight: 600;
          color: var(--ink);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 160px;
        }
        .qa-citation-popup-score {
          font-family: var(--font-pixel);
          font-size: 10px;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
          line-height: 1.3;
        }
        .qa-citation-popup-close {
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: transparent;
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          cursor: pointer;
          color: var(--ink-secondary);
          font-size: 10px;
          padding: 0;
          transition: transform 150ms var(--ease-base), box-shadow 150ms var(--ease-base), color 150ms var(--ease-base);
        }
        .qa-citation-popup-close:hover {
          transform: translate(-1px, -1px);
          box-shadow: 2px 2px 0 var(--ink);
          color: var(--brand);
        }
        .qa-citation-popup-close:active {
          transform: translate(0, 0);
          box-shadow: none;
        }
        .qa-citation-popup-body {
          padding: 10px 12px;
          font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
          font-size: 12px;
          line-height: 1.55;
          color: var(--ink-secondary);
        }
        .qa-citation-popup-preview .qa-citation-popup-body {
          display: -webkit-box;
          -webkit-line-clamp: 6;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .qa-citation-popup-pinned .qa-citation-popup-body {
          max-height: 320px;
          overflow-y: auto;
        }
        .qa-citation-popup-line {
          margin-bottom: 4px;
        }
        .qa-citation-popup-actions {
          display: flex;
          gap: 8px;
          padding: 10px 12px;
          border-top: 1px solid rgba(43,36,25,0.15);
          background: var(--bg-sunken);
        }

        /* Reasoning timeline */
        .qa-reasoning {
          border-bottom: 1px solid rgba(43,36,25,0.12);
          background: var(--bg-panel);
        }
        .qa-reasoning-summary {
          padding: 10px 12px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
          font-size: 13px;
          color: var(--ink);
          background: var(--bg-sunken);
          transition: background 150ms var(--ease-base);
        }
        .qa-reasoning-summary:hover {
          background: var(--bg-panel);
        }
        .qa-step-count {
          font-size: 11px;
          color: var(--ink-secondary);
          background: var(--bg-panel);
          padding: 1px 6px;
          border: 1px solid rgba(43,36,25,0.12);
          border-radius: var(--radius);
        }
        .qa-reasoning-steps {
          padding: 0 12px 12px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .qa-step {
          display: flex;
          gap: 12px;
          padding: 10px 12px;
          border: 1.5px solid var(--ink);
          border-radius: var(--radius);
          background: var(--bg-panel);
        }
        .qa-step-num {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: var(--bg-sunken);
          border: 1.5px solid var(--ink);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 11px;
          font-weight: 700;
          color: var(--ink);
          flex-shrink: 0;
        }
        .qa-step-body { flex: 1; min-width: 0; }
        .qa-step-action {
          font-size: 12px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 2px;
        }
        .qa-step-detail {
          font-size: 12px;
          color: var(--ink-secondary);
          word-break: break-all;
        }
        .qa-step-confidence {
          font-size: 10px;
          font-weight: 700;
          padding: 2px 6px;
          border-radius: var(--radius);
          height: fit-content;
          flex-shrink: 0;
          border: 1.5px solid var(--ink);
          color: #fff;
        }
        .qa-confidence-high { background: var(--cite-3); }
        .qa-confidence-medium { background: var(--cite-4); }
        .qa-confidence-low { background: var(--brand); }

        /* Feedback */
        .qa-feedback {
          position: relative;
          padding: 8px 12px;
          border-top: 1px solid rgba(43,36,25,0.15);
          background: var(--bg-sunken);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          overflow: visible;
        }
        .qa-feedback-note {
          font-size: 12px;
          color: var(--ink-secondary);
        }
        .qa-feedback-actions {
          display: flex;
          gap: 8px;
        }
        .qa-feedback-btn {
          color: var(--ink-secondary) !important;
          transition: transform 150ms var(--ease-base), box-shadow 150ms var(--ease-base), color 150ms var(--ease-base) !important;
        }
        .qa-feedback-btn:hover {
          color: var(--brand) !important;
          transform: translate(-1px, -1px);
          box-shadow: 3px 3px 0 var(--ink);
        }
        .qa-feedback-btn:active {
          transform: translate(2px, 2px) !important;
          box-shadow: none !important;
          transition: none !important;
        }
        .qa-feedback-stamp {
          position: absolute;
          right: 12px;
          top: 50%;
          transform: translateY(-50%) rotate(-12deg);
          width: 64px;
          height: 64px;
          border-radius: 50%;
          border: 2px solid var(--brand);
          color: var(--brand);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 13px;
          font-weight: 700;
          opacity: 0.85;
          pointer-events: none;
          background: rgba(200,57,43,0.06);
          animation: qa-feedback-stamp-appear 260ms var(--ease-pop) forwards;
        }
        @keyframes qa-feedback-stamp-appear {
          0% { transform: translateY(-50%) scale(1.4) rotate(-24deg); opacity: 0; }
          100% { transform: translateY(-50%) scale(1) rotate(-12deg); opacity: 0.85; }
        }

        /* Modal */
        .qa-feedback-modal .ant-modal-content {
          border: 1.5px solid var(--ink) !important;
          border-radius: var(--radius) !important;
          box-shadow: 6px 6px 0 var(--ink) !important;
          background: var(--bg-panel) !important;
        }
        .qa-feedback-modal .ant-modal-header {
          border-bottom: 1px solid var(--ink) !important;
          background: var(--bg-panel) !important;
          border-radius: var(--radius) var(--radius) 0 0 !important;
        }
        .qa-feedback-modal .ant-modal-title {
          color: var(--ink) !important;
          font-weight: 600;
        }
        .qa-feedback-select .ant-select-selector {
          border-radius: var(--radius) !important;
          border-color: var(--ink) !important;
          background: var(--bg-panel) !important;
        }
        .qa-feedback-select.ant-select-focused .ant-select-selector {
          border-color: var(--brand) !important;
          box-shadow: none !important;
        }
        .qa-feedback-textarea {
          border-radius: var(--radius) !important;
          border-color: var(--ink) !important;
          background: var(--bg-panel) !important;
        }
        .qa-feedback-textarea:focus {
          border-color: var(--brand) !important;
          box-shadow: none !important;
        }

        /* Replace radio cards */
        .qa-replace-cards {
          display: flex;
          flex-direction: column;
          gap: 8px;
          max-height: 320px;
          overflow-y: auto;
          padding-right: 4px;
        }
        .qa-replace-card {
          display: flex;
          align-items: flex-start;
          gap: 10px;
          padding: 10px 12px;
          border-radius: var(--radius);
          border: 1.5px solid transparent;
          background: var(--bg-panel);
          cursor: pointer;
          transition: background 150ms var(--ease-base), border-color 150ms var(--ease-base), box-shadow 150ms var(--ease-base);
        }
        .qa-replace-card:hover {
          border-color: rgba(43,36,25,0.25);
        }
        .qa-replace-card-selected {
          border-color: var(--ink) !important;
          box-shadow: 3px 3px 0 var(--ink);
        }
        .qa-replace-card-content {
          flex: 1;
          min-width: 0;
        }
        .qa-replace-card-meta {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
        }
        .qa-replace-card-file {
          font-size: 12px;
          font-weight: 600;
          color: var(--ink);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          flex: 1;
        }
        .qa-replace-card-score {
          font-size: 11px;
          font-weight: 700;
          font-family: 'Press Start 2P', 'JetBrains Mono', 'SF Mono', Consolas, monospace;
          font-variant-numeric: tabular-nums;
          flex-shrink: 0;
          line-height: 1.3;
        }
        .qa-replace-card-summary {
          font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
          font-size: 11px;
          line-height: 1.55;
          color: var(--ink-secondary);
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .qa-replace-citation-select {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }

        /* Flag modal */
        .qa-flag-radio-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .qa-flag-radio-group .ant-radio-wrapper {
          color: var(--ink);
        }

        /* Markdown overrides inside QA panel */
        .qa-answer-body .markdown-body code {
          background: var(--bg-sunken);
          color: var(--ink);
          padding: 2px 5px;
          border-radius: 2px;
          font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
          font-size: 0.9em;
        }
        .qa-answer-body .markdown-body pre {
          position: relative;
          background: var(--ink);
          color: var(--bg-paper);
          padding: 36px 14px 14px;
          border-radius: var(--radius);
          overflow-x: auto;
          margin: 0.6em 0;
          font-size: 12px;
          line-height: 1.55;
          border: 1.5px solid var(--ink);
        }
        .qa-answer-body .markdown-body pre::before {
          content: '';
          position: absolute;
          top: 12px;
          left: 12px;
          width: 10px;
          height: 10px;
          background: var(--brand);
          box-shadow: 16px 0 0 var(--cite-4), 32px 0 0 var(--cite-3);
        }
        .qa-answer-body .markdown-body pre code { background: transparent; color: inherit; padding: 0; }
        .qa-answer-body .markdown-body blockquote {
          border-left: 3px solid var(--brand);
          padding: 4px 12px;
          color: var(--ink-secondary);
          margin: 0.6em 0;
          background: var(--brand-soft);
        }
        .qa-answer-body .markdown-body table {
          border-collapse: collapse;
          width: 100%;
          margin: 0.6em 0;
          font-size: 13px;
          border: 1.5px solid var(--ink);
        }
        .qa-answer-body .markdown-body th, .qa-answer-body .markdown-body td {
          border: 1px solid rgba(43,36,25,0.15);
          padding: 8px 12px;
          text-align: left;
        }
        .qa-answer-body .markdown-body th { background: var(--bg-sunken); font-weight: 600; color: var(--ink); }

        @media (prefers-reduced-motion: reduce) {
          .qa-panel-root, .qa-panel-root * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 100ms !important;
          }
        }
      `}</style>
      {/* Slim toolbar: persona + kb + token + clear */}
      <div
        className="qa-toolbar"
        style={{
          padding: '8px 12px',
          marginBottom: 8,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          minHeight: 44,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0, flex: '0 1 auto' }}>
          <Tooltip title={personaName} placement="bottom">
            <span className="qa-persona-sticker">
              {personaName}
            </span>
          </Tooltip>
          <Tooltip title={collectionName} placement="bottom">
            <Select
              value={collectionName}
              onChange={onCollectionChange}
              options={collectionOptions}
              style={{ width: '100%' }}
              size="small"
              className="qa-kb-select"
            />
          </Tooltip>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
          {lastCost && (
            <>
              <span className="qa-token-round" style={{ fontSize: 11, color: 'var(--ink-secondary)' }}>
                本轮 <span className="qa-token-count" style={{ color: 'var(--ink)' }}>{lastCost.total_tokens.toLocaleString()}</span>
              </span>
              <span className="qa-token-round-only">
                <span className="qa-token-count" style={{ color: 'var(--ink)' }}>{lastCost.total_tokens.toLocaleString()}</span>
              </span>
            </>
          )}
          <Tooltip title={`距离上限还差 ${Math.max(SESSION_BUDGET - propsSessionTotal, 0).toLocaleString()} tokens`}>
            <div className="qa-token-bar">
              {Array.from({ length: tokenSegments }).map((_, i) => (
                <div
                  key={i}
                  className={`qa-token-segment ${i < usedSegments ? '' : 'empty'}`}
                />
              ))}
            </div>
          </Tooltip>
          <span className="qa-session-tokens" style={{ fontSize: 11, color: budgetColor, fontWeight: 600, minWidth: 82, textAlign: 'right' }}>
            <span className="qa-token-count">{propsSessionTotal.toLocaleString()}</span>
            <span className="qa-token-denom"> / {SESSION_BUDGET.toLocaleString()}</span>
          </span>
          <span className="qa-session-tokens-compact" style={{ color: budgetColor }}>
            <Database size={14} /> <span className="qa-token-count">{propsSessionTotal.toLocaleString()}</span>
          </span>
          <Tooltip title="清空对话">
            <Button
              icon={<Trash2 size={14} />}
              onClick={handleClear}
              disabled={messages.length === 0}
              size="small"
              type="text"
              style={{ color: 'var(--ink-secondary)' }}
            />
          </Tooltip>
        </div>
      </div>

      {/* Main workspace */}
      <div style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        overflow: 'hidden',
        border: '1.5px solid var(--ink)',
        borderRadius: 'var(--radius)',
        background: 'var(--bg-panel)',
      }}>
        {messages.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="qa-empty-state">
              <svg className="qa-empty-pencil" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="square" strokeLinejoin="miter">
                <path d="M12 19l7-7 3 3-7 7h-3v-3z" />
                <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                <path d="M2 2l7 7" />
                <path d="M15 5l4 4" />
              </svg>
              <span>这条线索断了，换个问法试试</span>
            </div>
            {composer}
          </div>
        ) : (
          <>
            {/* Left: retrieval panel for the last assistant message */}
            {(() => {
              const lastAssistant = messages[messages.length - 1];
              if (lastAssistant?.role !== 'assistant' || !lastAssistant.sources?.length) {
                return (
                  <div className={`qa-retrieval-panel${retrievalCollapsed ? ' collapsed' : ''}`}>
                    {retrievalCollapsed ? (
                      <button
                        className="qa-panel-toggle"
                        onClick={() => setRetrievalCollapsed(false)}
                        title="展开检索结果"
                      >
                        <ChevronRight size={18} />
                      </button>
                    ) : (
                      <>
                        <div className="qa-panel-header">
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                            <span className="qa-panel-title">找到的线索</span>
                            <Tooltip title="按来源范围过滤下方片段">
                              <QuestionCircleOutlined style={{ color: 'var(--ink-faint)', fontSize: 12 }} />
                            </Tooltip>
                            <Select
                              value="all"
                              disabled
                              options={[{ value: 'all', label: '全部' }]}
                              size="small"
                              className="qa-source-filter"
                              classNames={{ popup: { root: 'qa-source-filter-dropdown' } }}
                            />
                          </div>
                          <button
                            className="qa-panel-toggle"
                            onClick={() => setRetrievalCollapsed(true)}
                            title="收起检索结果"
                          >
                            <ChevronLeft size={12} />
                          </button>
                        </div>
                        <div className="qa-scroll">
                          <Text style={{ color: 'var(--ink-faint)', fontSize: 12, padding: 12, lineHeight: 1.6 }}>
                            {lastAssistant?.role !== 'assistant'
                              ? '等待助手回答…'
                              : '当前回答未引用任何检索片段（可能知识库不匹配或检索为空）'}
                          </Text>
                        </div>
                      </>
                    )}
                  </div>
                );
              }
              const clusters = clusterSources(lastAssistant.sources);
              return (
                <SourcePanel
                  sources={lastAssistant.sources}
                  clusters={clusters}
                  activeId={activeCitationId}
                  pinnedId={pinnedCitationId}
                  hoveredCiteIndex={hoveredCiteIndex}
                  pulsingSourceId={pulsingSourceId}
                  filterFile={sourceFilter}
                  onFilterChange={setSourceFilter}
                  onActivate={(id, globalIdx) => {
                    if (id && globalIdx) {
                      setHoveredCiteIndex(globalIdx);
                      setActiveCitationId(id);
                    } else {
                      setHoveredCiteIndex(null);
                      if (!pinnedCitationId) setActiveCitationId(undefined);
                    }
                  }}
                  onPin={setPinnedCitationId}
                  onRef={(id, el) => {
                    if (el) sourceRefs.current.set(id, el);
                    else sourceRefs.current.delete(id);
                  }}
                  onSourceBadgeClick={handleSourceBadgeClick}
                  panelCollapsed={retrievalCollapsed}
                  onToggle={() => setRetrievalCollapsed((v) => !v)}
                />
              );
            })()}

            {/* Right: answer area */}
            <div className="qa-answer-area">
              <div className="qa-scroll" style={{ padding: 16 }}>
                <AnimatePresence initial={false}>
                  {messages.map((msg, idx) => {
                    const isLastAssistant = idx === lastAssistantIdx && msg.role === 'assistant';
                    const clusters = msg.sources ? clusterSources(msg.sources) : [];
                    const citeProps = {
                      sources: msg.sources || [],
                      msgIdx: idx,
                      activeId: activeCitationId,
                      pinnedId: pinnedCitationId,
                      hoveredCiteIndex,
                      pulsingCitationIdx,
                      onDropReplace: isLastAssistant ? handleDropReplace : undefined,
                      citationOverrides: citationOverrides[idx] || {},
                      onRegisterCitation: registerCitation,
                      onUnregisterCitation: unregisterCitation,
                      onCitationEnter: handleCitationEnter,
                      onCitationLeave: handleCitationLeave,
                      onCitationClick: handleCitationClick,
                    };

                    if (msg.role === 'user') {
                      return (
                        <motion.div
                          key={idx}
                          style={{
                            display: 'flex',
                            justifyContent: 'flex-end',
                            marginBottom: 16,
                            gap: 10,
                          }}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.15, ease: [0.25, 0.8, 0.25, 1] }}
                        >
                          <div className="qa-user-bubble">
                            {msg.content}
                          </div>
                          <Avatar
                            icon={<User size={16} />}
                            style={{ background: 'var(--ink-faint)', flexShrink: 0 }}
                            size="small"
                          />
                        </motion.div>
                      );
                    }

                    if (msg.content.startsWith('请求失败：')) {
                      return (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.15 }}
                        >
                          <Alert
                            message={msg.content.replace('请求失败：', '')}
                            type="error"
                            showIcon
                            className="qa-error-alert"
                            style={{ marginBottom: 16 }}
                          />
                        </motion.div>
                      );
                    }

                    return (
                      <motion.div
                        key={idx}
                        className="qa-answer-card"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2, ease: [0.25, 0.8, 0.25, 1] }}
                      >
                        <div className="qa-answer-card-inner">
                        {msg.reasoning_steps && msg.reasoning_steps.length > 0 && (
                          <ReasoningTimeline steps={msg.reasoning_steps} />
                        )}

                        <div className="qa-answer-toolbar">
                          <Space size={8}>
                            <Bot size={16} style={{ color: 'var(--brand)' }} />
                            <span className="qa-answer-toolbar-label">
                              讲解员说
                            </span>
                            {msg.token_cost && (
                              <Tag style={{ margin: 0, fontSize: 10, borderColor: 'rgba(43,36,25,0.15)', color: 'var(--ink-secondary)', background: 'var(--bg-sunken)' }}>
                                {(msg.token_cost.total_tokens / 1000).toFixed(1)}k tokens
                              </Tag>
                            )}
                          </Space>
                          <Space size={8}>
                            {msg.sources && (
                              <span style={{ fontSize: 11, color: 'var(--ink-faint)' }}>
                                引用 {msg.sources.length} 条来源
                              </span>
                            )}
                          </Space>
                        </div>

                        <div className="qa-answer-body">
                          <div className="markdown-body">
                            <CitationPropsContext.Provider value={citeProps}>
                              <ReactMarkdown components={citationMarkdownComponents}>
                                {msg.content}
                              </ReactMarkdown>
                            </CitationPropsContext.Provider>
                          </div>
                        </div>

                        {isLastAssistant && (
                          <div className="qa-feedback">
                            <span className="qa-feedback-note">
                              <HelpCircle size={14} style={{ marginRight: 4, verticalAlign: 'text-bottom' }} />
                              发现引用或结论有问题？
                            </span>
                            <div className="qa-feedback-actions">
                              <Tooltip title="选择另一条检索片段替换当前引用">
                                <Button
                                  size="small"
                                  type="text"
                                  icon={<SwapOutlined />}
                                  className="qa-feedback-btn"
                                  onClick={() => openReplaceModal(idx)}
                                  loading={replaceSubmitting}
                                  disabled={flagSubmitting || replaceSubmitting}
                                >
                                  替换引用
                                </Button>
                              </Tooltip>
                              <Tooltip title="反馈引用不准确或存在问题">
                                <Button
                                  size="small"
                                  type="text"
                                  icon={<FlagOutlined />}
                                  className="qa-feedback-btn"
                                  onClick={() => openFlagModal(idx)}
                                  loading={flagSubmitting}
                                  disabled={flagSubmitting || replaceSubmitting || feedbackStamps[idx]}
                                >
                                  标记不准确
                                </Button>
                              </Tooltip>
                            </div>
                            {feedbackStamps[idx] && (
                              <div className="qa-feedback-stamp">已反馈</div>
                            )}
                          </div>
                        )}
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>

                {loading && (
                  <div className="qa-skeleton">
                    <Avatar icon={<Bot size={16} />} style={{ background: 'var(--ink-faint)', flexShrink: 0 }} size="small" />
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                      padding: 12,
                      borderRadius: 'var(--radius)',
                      background: 'var(--bg-panel)',
                      border: '1.5px solid var(--ink)',
                      width: 220,
                    }}>
                      <div className="qa-skeleton-block" style={{ width: '80%' }} />
                      <div className="qa-skeleton-block" style={{ width: '60%' }} />
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              {composer}
            </div>
          </>
        )}
      </div>

      {/* Floating citation popup */}
      <div ref={popupRef} style={popupStyle}>
        <CitationPopup
          source={popupSource}
          globalIdx={popupGlobalIdx}
          visible={popup.visible}
          pinned={popup.pinned}
          onMouseEnter={handlePopupEnter}
          onMouseLeave={handlePopupLeave}
          onClose={closePopup}
          onReplace={() => {
            if (popup.msgIdx !== null && popup.citationNum !== null) {
              closePopup();
              openReplaceModal(popup.msgIdx, popup.citationNum);
            }
          }}
          onInaccurate={() => {
            if (popup.msgIdx !== null && popup.citationNum !== null) {
              closePopup();
              openFlagModal(popup.msgIdx, popup.citationNum);
            }
          }}
        />
      </div>

      {/* Feedback modal */}
      <Modal
        className="qa-feedback-modal"
        open={feedbackType !== null}
        onCancel={() => {
          setFeedbackType(null);
          setFeedbackNote('');
          setReplaceTargetId('');
          setReplaceCitationNum(null);
          setFeedbackMsgIdx(null);
          setFlagProblemType(null);
          setFlagSubmitted(false);
          setReplaceSubmitting(false);
          setFlagSubmitting(false);
        }}
        footer={null}
        title={feedbackType === 'replace' ? (
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink)' }}>换一条线索</div>
            <div style={{ fontSize: 12, fontWeight: 400, color: 'var(--ink-secondary)', marginTop: 2 }}>从下方候选片段中选择一条替换当前引用</div>
          </div>
        ) : '标记不准确'}
        width={520}
      >
        {feedbackType === 'replace' && (
          <div>
            <Text style={{ fontSize: 13, color: 'var(--ink-secondary)', display: 'block', marginBottom: 12 }}>
              选择更合适的来源片段。该反馈会保存到当前会话，并在后续回答中作为事实修正约束生效。
            </Text>
            <div className="qa-replace-citation-select">
              <Text style={{ fontSize: 12, color: 'var(--ink)' }}>要替换的引用：</Text>
              <Select
                size="small"
                className="qa-feedback-select"
                value={replaceCitationNum ?? 1}
                onChange={(val) => setReplaceCitationNum(val)}
                options={(() => {
                  const msg = feedbackMsgIdx !== null ? messages[feedbackMsgIdx] : null;
                  return (msg?.sources || []).map((_, i) => ({ value: i + 1, label: `[${i + 1}]` }));
                })()}
                style={{ width: 80 }}
              />
            </div>
            <Radio.Group
              value={replaceTargetId || undefined}
              onChange={(e) => setReplaceTargetId(e.target.value)}
              className="qa-replace-cards"
            >
              {(() => {
                const msg = feedbackMsgIdx !== null ? messages[feedbackMsgIdx] : null;
                const selectedSourceId = replaceTargetId;
                return (msg?.sources || []).map((s, i) => {
                  const globalIdx = i + 1;
                  const selected = selectedSourceId === s.id;
                  const summary = getSummaryLines(s.content, 3);
                  return (
                    <div
                      key={s.id}
                      className={`qa-replace-card ${selected ? 'qa-replace-card-selected' : ''}`}
                      style={selected ? { background: colorWithOpacity(getCiteColorVar(globalIdx), 0.08) } : undefined}
                      onClick={() => setReplaceTargetId(s.id)}
                    >
                      <Radio value={s.id} />
                      <div className="qa-replace-card-content">
                        <div className="qa-replace-card-meta">
                          <span
                            className="qa-source-id"
                            style={{ background: getCiteColorVar(globalIdx), width: 18, height: 18, minWidth: 18, fontSize: 9 }}
                          >
                            {globalIdx}
                          </span>
                          <span className="qa-replace-card-file" title={s.file_name}>{s.file_name}</span>
                          <span className="qa-replace-card-score" style={{ color: scoreColorClass(s.similarity) }}>
                            {(s.similarity * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="qa-replace-card-summary">
                          {summary.map((line, li) => <div key={li}>{line}</div>)}
                        </div>
                      </div>
                    </div>
                  );
                });
              })()}
            </Radio.Group>
            <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <Button onClick={() => { setFeedbackType(null); setFeedbackNote(''); setReplaceTargetId(''); setReplaceCitationNum(null); setFeedbackMsgIdx(null); setReplaceSubmitting(false); }}>取消</Button>
              <Button type="primary" className="qa-send-btn" icon={<Check size={14} />} onClick={confirmReplace} disabled={!replaceTargetId} loading={replaceSubmitting}>
                确认替换
              </Button>
            </div>
          </div>
        )}
        {feedbackType === 'inaccurate' && (
          <div>
            <Text style={{ fontSize: 13, color: 'var(--ink-secondary)', display: 'block', marginBottom: 12 }}>
              请选择问题类型并简要说明。该反馈会保存到当前会话，并在后续回答中作为事实修正约束生效。
            </Text>
            <Radio.Group
              value={flagProblemType}
              onChange={(e) => setFlagProblemType(e.target.value)}
              className="qa-flag-radio-group"
              style={{ marginBottom: 12 }}
            >
              {FLAG_OPTIONS.map((opt) => (
                <Radio key={opt.value} value={opt.value}>{opt.label}</Radio>
              ))}
            </Radio.Group>
            <TextArea
              className="qa-feedback-textarea"
              rows={4}
              value={feedbackNote}
              onChange={(e) => setFeedbackNote(e.target.value)}
              placeholder="补充说明（可选，最多 200 字）"
              maxLength={200}
              showCount
            />
            <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <Button onClick={() => { setFeedbackType(null); setFeedbackNote(''); setFlagProblemType(null); setFlagSubmitted(false); setFeedbackMsgIdx(null); setFlagSubmitting(false); }}>取消</Button>
              <Button
                type="primary"
                danger
                className="qa-send-btn"
                icon={<Check size={14} />}
                onClick={submitInaccurate}
                disabled={!flagProblemType || flagSubmitted}
                loading={flagSubmitting}
              >
                {flagSubmitted ? '已反馈' : '提交反馈'}
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
