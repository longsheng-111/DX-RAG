'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Button, Input, Progress, Typography, Space, Spin, Alert, Divider,
} from 'antd';
import TextareaAutosize from 'react-textarea-autosize';
import api from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import css from 'styled-jsx/css';
import {
  Play, RotateCcw, Send, Flag, CheckCircle2, AlertTriangle, Target,
} from 'lucide-react';

const { Title, Text } = Typography;

interface ExamPoint {
  point: string;
  hit: boolean;
  evidence: string;
}

interface ExamScore {
  question: string;
  answer: string;
  score: number;
  feedback: string;
  points?: ExamPoint[];
  missed_points?: string[];
}

interface ExamSummary {
  raw: string;
  total_score: number;
}

interface ExamEvaluation {
  raw: string;
  score: number;
  points?: ExamPoint[];
  comment?: string;
  supplement?: string;
  correction?: string;
}

interface ExamState {
  session_id: string;
  status: 'asking' | 'follow_up' | 'finished';
  question_index: number;
  current_question: string;
  current_expectations: string[];
  reference_points: string[];
  follow_up_count: number;
  evaluation?: ExamEvaluation;
  summary?: ExamSummary;
  scores: ExamScore[];
  weak_points: string[];
  cheating_detected?: boolean;
}

interface Props {
  sessionId: string;
  collectionName: string;
}

const MAX_QUESTIONS = 5;

/* ============================================================
   教室叙事 · 粉笔盒（v1.8 拟真重绘）
   三点透视精细手绘 SVG，分层：盒身/盒盖/粉笔×N/灰堆
   结构真实但仍带 1.5px 描边的插画感，不做照片级渐变
   ============================================================ */
function ChalkBox({ wholeCount, stubCount, animStage }: {
  wholeCount: number;
  stubCount: number;
  animStage: 'idle' | 'start' | 'score';
}) {
  const total = wholeCount + stubCount;
  const chalkColors = ['var(--chalk-bright)', 'var(--chalk-yellow)', 'var(--brand)', 'var(--chalk-bright)', 'var(--chalk-bright)'];
  // 粉笔在盒内的 x 位置（沿盒口前缘分布）
  const chalkPositions = [
    { cx: 68, baseY: 56 } as const,
    { cx: 83, baseY: 55 } as const,
    { cx: 98, baseY: 57 } as const,
    { cx: 113, baseY: 54 } as const,
    { cx: 128, baseY: 56 } as const,
  ];

  return (
    <svg
      className={`chalk-box chalk-box--${animStage}`}
      viewBox="0 0 200 130"
      width="110"
      height="72"
      aria-label="粉笔盒"
      style={{ position: 'absolute', left: 8, bottom: -24, zIndex: 1 }}
    >
      {/* ---- 盒底投影（硬阴影语言，按透视变形） ---- */}
      <polygon points="42,112 168,112 178,86 48,86"
        fill="rgba(0,0,0,0.18)"
      />

      {/* ---- 盒身右侧面（背光面，加深 12%） ---- */}
      <polygon points="157,50 162,108 172,80 167,24"
        fill="#9B8B72" stroke="var(--ink)" strokeWidth="1.5" strokeLinejoin="round"
      />

      {/* ---- 盒身正面 ---- */}
      <polygon points="37,108 162,108 157,50 42,50"
        fill="#C4B08A" stroke="var(--ink)" strokeWidth="1.5" strokeLinejoin="round"
      />

      {/* ---- 盒口内腔（深色，深于板色 15%） ---- */}
      <polygon points="42,50 157,50 167,24 52,24"
        fill="#1A3025" stroke="var(--ink)" strokeWidth="1.5" strokeLinejoin="round"
      />
      {/* 盒口前缘纸板厚度线 */}
      <line x1="42" y1="50" x2="157" y2="50"
        stroke="var(--ink)" strokeWidth="1.2" />

      {/* ---- 盒盖（翻开立起于盒身侧后方，内面浅纸板色） ---- */}
      <g className="chalk-box-lid">
        {/* 盒盖内面 */}
        <polygon points="52,24 167,24 148,-8 33,-8"
          fill="#DDD0B8" stroke="var(--ink)" strokeWidth="1.5" strokeLinejoin="round"
        />
        {/* 盒盖纸板厚度边 */}
        <line x1="52" y1="24" x2="167" y2="24"
          stroke="var(--ink)" strokeWidth="1.2" />
      </g>

      {/* ---- 粉笔（圆柱体 + 顶面椭圆高光） ---- */}
      {Array.from({ length: total }).map((_, i) => {
        const isStub = i < stubCount;
        const pos = chalkPositions[i] || chalkPositions[chalkPositions.length - 1];
        const fullHeight = 45 - (i % 3) * 4; // 每支高度略有变化
        const stubHeight = 10 + (i % 3) * 2;
        const chalkH = isStub ? stubHeight : fullHeight;
        const topY = pos.baseY - chalkH;
        const cls = `chalk-stick${isStub ? ' chalk-stick--stub' : ''}${i === stubCount ? ' chalk-stick--first' : ''}`;
        return (
          <g key={i} className={cls}>
            {/* 粉笔柱身 */}
            <rect x={pos.cx - 5} y={topY} width="10" height={chalkH} rx="3"
              fill={chalkColors[i % chalkColors.length]}
              stroke="var(--ink)" strokeWidth="0.8"
            />
            {/* 顶面椭圆高光 */}
            <ellipse cx={pos.cx} cy={topY} rx="5" ry="2.5"
              fill="rgba(255,255,255,0.45)"
              stroke="var(--ink)" strokeWidth="0.6"
            />
          </g>
        );
      })}

      {/* ---- 起跳粉笔（animStage==='start' 时跳出盒口） ---- */}
      {animStage === 'start' && (
        <g className="chalk-jumper">
          <rect x="93" y="4" width="10" height="46" rx="3"
            fill="var(--chalk-bright)" stroke="var(--ink)" strokeWidth="0.8"
          />
          <ellipse cx="98" cy="4" rx="5" ry="2.5"
            fill="rgba(255,255,255,0.45)" stroke="var(--ink)" strokeWidth="0.6"
          />
        </g>
      )}

      {/* ---- 盒身正面印刷 ---- */}
      {/* 品牌红装饰横条 */}
      <rect x="52" y="72" width="92" height="2.5" rx="1"
        fill="var(--brand)" opacity="0.85" />
      {/* DX 无尘粉笔 */}
      <text x="98" y="88" textAnchor="middle"
        fontFamily="var(--font-display)" fontSize="7.5" fontWeight="600"
        fill="var(--ink)" letterSpacing="0.5">
        DX 无尘粉笔
      </text>
      {/* 角落小粉笔图案 */}
      <g transform="translate(135, 96) scale(0.6)" opacity="0.5">
        <rect x="0" y="0" width="6" height="18" rx="2"
          fill="none" stroke="var(--ink)" strokeWidth="1.5" />
        <ellipse cx="3" cy="0" rx="3" ry="1.5"
          fill="none" stroke="var(--ink)" strokeWidth="1" />
      </g>

      {/* ---- 盒身侧面竖排小字 ---- */}
      <text x="167" y="68" textAnchor="middle"
        fontFamily="var(--font-display)" fontSize="5.5" fill="var(--ink)" opacity="0.55"
        writingMode="vertical-rl" letterSpacing="1">
        5支装
      </text>

      {/* ---- 粉笔灰粒子（盒底与槽面接触处） ---- */}
      <circle cx="52" cy="113" r="1.5" fill="var(--chalk)" opacity="0.45" />
      <circle cx="72" cy="115" r="1.2" fill="var(--chalk)" opacity="0.35" />
      <circle cx="90" cy="114" r="1.0" fill="var(--chalk-yellow)" opacity="0.4" />
      <circle cx="110" cy="116" r="1.4" fill="var(--chalk)" opacity="0.35" />
      <circle cx="130" cy="113" r="1.1" fill="var(--chalk-bright)" opacity="0.4" />
      <circle cx="148" cy="115" r="1.3" fill="var(--chalk)" opacity="0.3" />
      <circle cx="160" cy="112" r="0.9" fill="var(--chalk)" opacity="0.35" />
      {/* 灰色粉痕 */}
      <ellipse cx="98" cy="114" rx="28" ry="2.5"
        fill="var(--chalk-faint)" opacity="0.18" />

      {/* ---- 评分阶段粉笔灰飘落动画 ---- */}
      {animStage === 'score' && (
        <>
          <circle cx="48" cy="108" r="1.2" fill="var(--chalk)" opacity="0.5">
            <animate attributeName="cy" from="108" to="118" dur="2s" fill="freeze" />
            <animate attributeName="opacity" from="0.5" to="0" dur="2s" fill="freeze" />
          </circle>
          <circle cx="145" cy="106" r="1" fill="var(--chalk)" opacity="0.4">
            <animate attributeName="cy" from="106" to="116" dur="1.8s" fill="freeze" />
            <animate attributeName="opacity" from="0.4" to="0" dur="1.8s" fill="freeze" />
          </circle>
          <circle cx="98" cy="108" r="0.8" fill="var(--chalk-yellow)" opacity="0.45">
            <animate attributeName="cy" from="108" to="120" dur="2.2s" fill="freeze" />
            <animate attributeName="opacity" from="0.45" to="0" dur="2.2s" fill="freeze" />
          </circle>
        </>
      )}
    </svg>
  );
}

/** 板面左上角花：五角星 + 三道光芒（粉笔简笔画·书写动画） */
function CornerStar() {
  return (
    <svg viewBox="0 0 64 64" width="56" height="56"
      style={{ position: 'absolute', left: 10, top: 10, opacity: 0.5, pointerEvents: 'none' }}
      aria-hidden="true"
    >
      <polygon points="32,8 38,26 56,26 42,38 48,56 32,46 16,56 22,38 8,26 26,26"
        fill="none" stroke="var(--chalk)" strokeWidth="1.5" strokeLinejoin="round"
        className="ep-write-path" style={{ animationDelay: '0ms' }} />
      <line x1="32" y1="56" x2="32" y2="64" stroke="var(--chalk)" strokeWidth="1" opacity="0.6"
        className="ep-write-path" style={{ animationDelay: '400ms' }} />
      <line x1="8" y1="26" x2="0" y2="22" stroke="var(--chalk)" strokeWidth="1" opacity="0.4"
        className="ep-write-path" style={{ animationDelay: '500ms' }} />
      <line x1="56" y1="26" x2="64" y2="22" stroke="var(--chalk)" strokeWidth="1" opacity="0.4"
        className="ep-write-path" style={{ animationDelay: '600ms' }} />
    </svg>
  );
}

/** 板面右下角花：台灯（粉笔简笔画·书写动画） */
function CornerLamp() {
  return (
    <svg viewBox="0 0 64 64" width="56" height="56"
      style={{ position: 'absolute', right: 12, bottom: 40, opacity: 0.5, pointerEvents: 'none' }}
      aria-hidden="true"
    >
      <ellipse cx="32" cy="52" rx="20" ry="3" fill="none" stroke="var(--chalk)" strokeWidth="1.5"
        className="ep-write-path" style={{ animationDelay: '0ms' }} />
      <line x1="16" y1="52" x2="22" y2="28" stroke="var(--chalk)" strokeWidth="1.2"
        className="ep-write-path" style={{ animationDelay: '300ms' }} />
      <line x1="48" y1="52" x2="42" y2="28" stroke="var(--chalk)" strokeWidth="1.2"
        className="ep-write-path" style={{ animationDelay: '400ms' }} />
      <path d="M22,28 Q32,8 42,28" fill="none" stroke="var(--chalk)" strokeWidth="1.5" strokeLinecap="round"
        className="ep-write-path" style={{ animationDelay: '500ms' }} />
      <line x1="22" y1="28" x2="42" y2="28" stroke="var(--chalk)" strokeWidth="1"
        className="ep-write-path" style={{ animationDelay: '700ms' }} />
      <circle cx="32" cy="22" r="2" fill="var(--chalk-yellow)" opacity="0.7"
        className="ep-write-dot" style={{ animationDelay: '900ms' }} />
    </svg>
  );
}

/** 粉笔盒与角花共享样式 → 移入 globals.css .ep-root 段 */

/* ============================================================
   教室叙事 · 毛毡公告栏（v1.8·批次二）
   左侧公告栏：课表 + 学习目标便签 + 纸飞机（3 层纵向构图）
   右侧公告栏：倒计时 + 小红花奖状 + 备忘清单（3 层纵向构图）
   ============================================================ */

/** 左侧公告栏：今日课表 → 学习目标便签 → 纸飞机 */
function LeftBulletinBoard({ isExam }: { isExam?: boolean }) {
  return (
    <div className={`ep-bulletin-board ep-bulletin-board--left${isExam ? ' ep-bulletin-board--exam-left' : ''}`} aria-hidden="true">
      <div className="ep-board-hanger" />
      <div className="ep-bulletin-board-inner">

      {/* 第 1 层：今日课表（升级：140-160px 宽，红色图钉） */}
      <div className="ep-board-item ep-board-item--schedule ep-board-item--first"
        style={{ transform: 'rotate(-1.5deg)' }}>
        <div className="ep-pushpin" />
        <div className="ep-schedule-card">
          <div className="ep-board-item-inner">
            <div className="ep-schedule-title">今日课表</div>
            <table className="ep-schedule-table">
              <tbody>
                <tr><td>第1节</td><td>概念题</td></tr>
                <tr><td>第2节</td><td>原理题</td></tr>
                <tr><td>第3节</td><td>对比题</td></tr>
                <tr><td>第4节</td><td>场景题</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 第 2 层：学习目标便签（黄色便签 + 3 行手写清单，前两行已勾选） */}
      <div className="ep-board-item ep-board-item--goals"
        style={{ transform: 'rotate(-1.5deg)' }}>
        <div className="ep-goals-sticky">
          <div className="ep-board-item-inner">
            <div className="ep-goals-title">📋 学习目标</div>
            <div className="ep-checklist-item ep--checked">掌握核心概念</div>
            <div className="ep-checklist-item ep--checked">理解原理机制</div>
            <div className="ep-checklist-item">能独立表达</div>
          </div>
        </div>
      </div>

      {/* 第 3 层：纸飞机（折纸飞机 + 虚线尾迹） */}
      <div className="ep-board-item ep-board-item--plane"
        style={{ transform: 'rotate(2deg)' }}>
        <div className="ep-plane-wrapper">
          {/* 纸飞机本体 */}
          <svg className="ep-plane-svg" viewBox="0 0 60 32" width="60" height="32" aria-hidden="true">
            <polygon points="30,2 50,24 36,20 30,30 24,20 10,24"
              fill="var(--bg-panel)" stroke="var(--ink)" strokeWidth="1.5"
              strokeLinejoin="round" />
            <line x1="30" y1="30" x2="30" y2="20"
              stroke="var(--ink)" strokeWidth="1" opacity="0.5" />
          </svg>
          {/* 尾迹虚线弧 */}
          <svg className="ep-plane-trail" viewBox="0 0 56 10" width="56" height="10" aria-hidden="true">
            <path d="M54,5 Q36,-2 18,5 Q10,7 2,5"
              fill="none" stroke="var(--ink-faint)" strokeWidth="1"
              strokeDasharray="2 3" strokeLinecap="round" opacity="0.5" />
          </svg>
        </div>
      </div>
      </div>{/* /ep-bulletin-board-inner */}
    </div>
  );
}

/** 右侧公告栏：倒计时便利贴 → 小红花奖状 → 备忘清单 */
function RightBulletinBoard({ isExam }: { isExam?: boolean }) {
  return (
    <div className={`ep-bulletin-board ep-bulletin-board--right${isExam ? ' ep-bulletin-board--exam-right' : ''}`} aria-hidden="true">
      <div className="ep-board-hanger" />
      <div className="ep-bulletin-board-inner">

      {/* 第 1 层：倒计时便利贴（升级：同课表量级，红色图钉） */}
      <div className="ep-board-item ep-board-item--countdown ep-board-item--first"
        style={{ transform: 'rotate(2deg)' }}>
        <div className="ep-pushpin" />
        <div className="ep-countdown-sticky">
          <div className="ep-board-item-inner">
            <div className="ep-countdown-label">距离面试还有</div>
            <div className="ep-countdown-number">? 天</div>
            <div className="ep-countdown-note">认真准备 · 全力以赴</div>
          </div>
        </div>
      </div>

      {/* 第 2 层：小红花奖状（迷你奖状贴纸，胶带贴角） */}
      <div className="ep-board-item ep-board-item--cert"
        style={{ transform: 'rotate(1.5deg)' }}>
        {/* 胶带贴角 */}
        <div className="ep-cert-tape" style={{ top: -3, left: 16 }} />
        <div className="ep-cert-tape" style={{ bottom: -3, right: 14 }} />
        <div className="ep-certificate">
          <div className="ep-board-item-inner">
            {/* 五角星 */}
            <svg viewBox="0 0 24 24" width="20" height="20"
              style={{ display: 'block', margin: '0 auto 2px' }} aria-hidden="true">
              <polygon points="12,2 15,9 22,9 16,14 18,21 12,17 6,21 8,14 2,9 9,9"
                fill="none" stroke="var(--brand)" strokeWidth="1.5"
                strokeLinejoin="round" />
            </svg>
            <div className="ep-cert-title">进步之星</div>
            <div className="ep-cert-sub">特发此状 · 以资鼓励</div>
          </div>
        </div>
      </div>

      {/* 第 3 层：备忘清单（纸色长条便签 + 趣味备忘） */}
      <div className="ep-board-item ep-board-item--memo"
        style={{ transform: 'rotate(-1deg)' }}>
        <div className="ep-memo-sticky">
          <div className="ep-board-item-inner">
            <div className="ep-memo-title">📝 备忘</div>
            <ul className="ep-memo-list">
              <li>背八股</li>
              <li>喝水</li>
              <li>别慌</li>
            </ul>
          </div>
        </div>
      </div>
      </div>{/* /ep-bulletin-board-inner */}
    </div>
  );
}

function scoreColor(score: number) {
  // 黑板场景：品牌红不上绿底，改用粉笔色阶
  if (score >= 7) return 'var(--cite-3)';
  if (score >= 5) return 'var(--cite-4)';
  return 'var(--chalk-faint)';
}

function ReferencePointsCard({ points }: { points: string[] }) {
  return (
    <div className="op-hint-box" style={{ marginBottom: 12 }}>
      <Text strong style={{ fontSize: 12, color: 'var(--chalk-yellow)', letterSpacing: 0.5 }}>参考答案要点</Text>
      <ol style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 13, color: 'var(--chalk)', lineHeight: 1.7 }}>
        {points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ol>
    </div>
  );
}

function PointsResultCard({ points }: { points?: ExamPoint[] }) {
  if (!points || points.length === 0) return null;
  const hitCount = points.filter((p) => p.hit).length;
  return (
    <div style={{ marginTop: 14 }}>
      <Text strong style={{ fontSize: 12, color: 'var(--chalk-yellow)', letterSpacing: 0.5 }}>
        要点命中 {hitCount} / {points.length}
      </Text>
      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {points.map((p, i) => (
          <div
            key={i}
            className="op-point"
            style={{
              borderColor: p.hit ? 'var(--cite-3)' : 'var(--chalk-yellow)',
            }}
          >
            <Space size={6}>
              {p.hit ? (
                <CheckCircle2 size={14} color="var(--cite-3)" />
              ) : (
                <AlertTriangle size={14} color="var(--chalk-yellow)" />
              )}
              <Text style={{ color: p.hit ? 'var(--chalk)' : 'var(--chalk-yellow)', fontWeight: 500 }}>
                {p.hit ? '命中' : '未命中'}
              </Text>
            </Space>
            <div style={{ marginTop: 4, color: 'var(--chalk)' }}>{p.point}</div>
            {p.evidence && (
              <div style={{ marginTop: 4, color: 'var(--chalk-dim)', fontSize: 11 }}>
                依据：{p.evidence}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** 判断 raw 文本是否为 JSON 裸奔（含 JSON 结构但无结构化字段已解析） */
function isRawJsonLike(raw: string): boolean {
  const s = raw.trim();
  return (s.startsWith('{') && s.includes('"points"')) || (s.startsWith('```json') && s.includes('"points"'));
}

/** 评分反馈结构化卡片（P0-1 修复） */
function EvaluationCard({ evaluation, onRetry }: { evaluation: ExamEvaluation; onRetry?: () => void }) {
  const hasStructured = !!(evaluation.comment || evaluation.supplement || evaluation.correction);

  // 结构化渲染
  if (hasStructured) {
    return (
      <>
        {evaluation.comment && (
          <div className="ep-eval-section">
            <div className="ep-eval-label">点评</div>
            <div className="markdown-body ep-eval-body">
              <ReactMarkdown>{evaluation.comment}</ReactMarkdown>
            </div>
          </div>
        )}
        {evaluation.supplement && (
          <div className="ep-eval-section">
            <div className="ep-eval-label">补充</div>
            <div className="markdown-body ep-eval-body">
              <ReactMarkdown>{evaluation.supplement}</ReactMarkdown>
            </div>
          </div>
        )}
        {evaluation.correction && (
          <div className="ep-eval-section">
            <div className="ep-eval-label">纠正</div>
            <div className="markdown-body ep-eval-body">
              <ReactMarkdown>{evaluation.correction}</ReactMarkdown>
            </div>
          </div>
        )}
        <PointsResultCard points={evaluation.points} />
      </>
    );
  }

  // 降级：raw 为 JSON 裸奔 → 错误卡
  if (isRawJsonLike(evaluation.raw)) {
    return (
      <Alert
        className="ep-alert"
        message="评分数据解析失败"
        description="后端返回了非结构化评分数据，请重试或联系管理员。"
        type="error"
        showIcon
        style={{
          background: 'rgba(200,57,43,0.12)',
          border: '1.5px solid var(--chalk-bright)',
          borderRadius: 3,
        }}
        action={
          onRetry && (
            <Button className="op-btn" size="small" onClick={onRetry}>
              重试
            </Button>
          )
        }
      />
    );
  }

  // 降级：raw 为旧格式纯文本
  return (
    <>
      <div className="markdown-body" style={{ fontSize: 14, color: 'var(--chalk)' }}>
        <ReactMarkdown>{evaluation.raw}</ReactMarkdown>
      </div>
      <PointsResultCard points={evaluation.points} />
    </>
  );
}

// 配置页与考试页共享的样式（原两段 styled-jsx 去重合并）。
// 考试页卡片独有的悬停上浮拆为 op-card-lift 修饰类；配置页头部内边距经
// .op-card-header.ep-config-header 复合选择器保留 14px 24px，两页渲染与合并前一致。
const examinerStyles = css`
  /* ===== 根容器：黑板 + 木框 + 粉笔槽（纯 CSS） ===== */
  .ep-root {
    position: relative;
    color: var(--chalk);
    background: var(--board);
    border: 8px solid var(--ink);
    border-bottom-width: 30px;
    box-shadow: inset 0 0 0 1.5px var(--chalk-weak);
  }
  /* 粉笔槽凹槽暗线（边框底部内侧） */
  .ep-root::before {
    content: '';
    position: absolute;
    left: 24px;
    bottom: -18px;
    width: 54px;
    height: 9px;
    border-radius: 4px;
    background: var(--chalk-bright);
    box-shadow:
      64px 0 0 var(--chalk-yellow),
      0 0 0 1px rgba(0,0,0,0.35),
      64px 0 0 1px rgba(0,0,0,0.35);
  }
  /* 板擦 */
  .ep-root::after {
    content: '';
    position: absolute;
    right: 32px;
    bottom: -21px;
    width: 28px;
    height: 14px;
    border-radius: 2px;
    background: var(--ink-secondary);
    box-shadow: 0 0 0 1px rgba(0,0,0,0.45);
  }
  .ep-root::before,
  .ep-root::after {
    filter: drop-shadow(0 1px 0 rgba(0,0,0,0.3));
  }

  /* ===== on-board 容器：重置文字上下文，杜绝 --ink 继承进黑板 ===== */
  .ep-root,
  .ep-root :global(*) {
    /* 不强制 * 的子，只对直接文字容器生效；markdown 走下方全局规则 */
  }

  /* ===== 粉笔卡片（v1.5 统一制式） ===== */
  .op-card {
    background: var(--board-deep);
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    background-image:
      repeating-linear-gradient(0deg, var(--chalk-weak) 0 1px, transparent 1px 24px),
      repeating-linear-gradient(90deg, var(--chalk-weak) 0 1px, transparent 1px 24px);
  }
  .op-card-lift {
    transition: transform 150ms cubic-bezier(0.25, 0.8, 0.25, 1),
      box-shadow 150ms cubic-bezier(0.25, 0.8, 0.25, 1);
  }
  .op-card-lift:hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 var(--chalk-dim);
  }
  .op-card-header {
    padding: 14px 16px;
    border-bottom: 1px solid var(--chalk-weak);
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 600;
    color: var(--chalk-bright);
  }
  .op-card-header.ep-config-header {
    padding: 14px 24px;
    font-family: var(--font-display);
    font-size: 23px;
    letter-spacing: 0.5px;
  }
  .ep-config-sub {
    margin-top: 2px;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: normal;
    color: var(--chalk-dim);
  }

  /* ===== 粉笔静态标签 ===== */
  .ep-chalk-note {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 12px;
    color: var(--chalk);
  }

  /* ===== 粉笔表单控件 ===== */
  .op-input {
    border: 1.5px solid var(--chalk-dim) !important;
    border-radius: 3px !important;
    background: transparent !important;
    color: var(--chalk-bright) !important;
    transition: border-color 150ms cubic-bezier(0.25, 0.8, 0.25, 1);
  }
  .op-input::placeholder { color: var(--chalk-dim) !important; }
  .op-input:focus {
    border-color: var(--chalk-dim) !important;
    outline: 2px solid var(--chalk-yellow);
    outline-offset: 2px;
  }
  .ep-root :global(.ant-input) {
    background: transparent !important;
    border-color: var(--chalk-dim) !important;
    color: var(--chalk-bright) !important;
  }
  .ep-root :global(.ant-input::placeholder) { color: var(--chalk-dim) !important; }
  .ep-root :global(.ant-input:hover) { border-color: var(--chalk-bright) !important; }
  .ep-root :global(.ant-input:focus) {
    border-color: var(--chalk-dim) !important;
    outline: 2px solid var(--chalk-yellow);
    outline-offset: 2px;
    box-shadow: none !important;
  }

  /* ===== 题目文字 ===== */
  .ep-question {
    font-size: 17px;
    line-height: 1.7;
    color: var(--chalk-bright);
    overflow-wrap: anywhere;
    word-break: break-word;
    max-width: 100%;
  }
  .ep-question p { color: var(--chalk-bright); overflow-wrap: anywhere; }

  /* ===== 考察标签 chip ===== */
  .ep-expectation-tag {
    display: inline-flex;
    align-items: center;
    max-width: 260px;
    height: 22px;
    padding: 0 8px;
    background: rgba(0,0,0,0.15);
    color: var(--chalk);
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ep-expectation-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
    max-height: 56px;
    overflow: hidden;
    position: relative;
  }

  /* ===== 标签/徽标 ===== */
  .op-tag {
    display: inline-flex;
    align-items: center;
    height: 22px;
    padding: 0 8px;
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 12px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }
  .op-tag-sunken {
    display: inline-flex;
    align-items: center;
    height: 22px;
    padding: 0 8px;
    background: rgba(0,0,0,0.15);
    color: var(--chalk);
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 12px;
    font-weight: 500;
  }

  /* ===== 提示框（参考答案要点） ===== */
  .op-hint-box {
    padding: 14px 16px;
    background: var(--board-deep);
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
  }

  /* ===== 文本域 ===== */
  .op-textarea {
    width: 100%;
    padding: 12px;
    background: var(--board-deep);
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 15px;
    line-height: 1.6;
    resize: none;
    outline: none;
    color: var(--chalk-bright);
    transition: border-color 150ms cubic-bezier(0.25, 0.8, 0.25, 1);
  }
  .op-textarea:focus {
    border-color: var(--chalk-yellow);
    outline: 2px solid var(--chalk-yellow);
    outline-offset: 2px;
  }
  .op-textarea::placeholder { color: var(--chalk-dim); }

  /* ===== 按钮体系 ===== */
  .op-btn {
    border-radius: 3px;
    border: 1.5px solid var(--chalk-dim);
    background: var(--board-deep);
    color: var(--chalk);
    transition: transform 150ms cubic-bezier(0.25, 0.8, 0.25, 1),
      box-shadow 150ms cubic-bezier(0.25, 0.8, 0.25, 1),
      border-color 150ms cubic-bezier(0.25, 0.8, 0.25, 1),
      background 150ms cubic-bezier(0.25, 0.8, 0.25, 1),
      color 150ms cubic-bezier(0.25, 0.8, 0.25, 1);
  }
  .ep-root :global(.ant-btn).op-btn {
    background: var(--board-deep) !important;
    border-color: var(--chalk-dim) !important;
    color: var(--chalk) !important;
  }
  .op-btn:hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 var(--chalk-dim);
    background: rgba(0,0,0,0.1);
    border-color: var(--chalk-yellow);
    color: var(--chalk-yellow);
  }
  .op-btn:active {
    transform: translate(0, 0);
    box-shadow: none;
  }
  .op-btn-primary {
    background: var(--brand);
    border-color: var(--chalk-dim);
    color: #fff;
  }
  .ep-root :global(.ant-btn).op-btn-primary {
    background: var(--brand) !important;
    border-color: var(--chalk-dim) !important;
    color: #fff !important;
  }
  .op-btn-primary:hover { background: var(--brand-hover); }
  .op-btn-primary:disabled {
    background: rgba(0,0,0,0.12);
    color: var(--chalk-faint);
    border-color: var(--chalk-faint);
  }
  .ep-root :global(.ant-btn).op-btn-primary:disabled {
    background: rgba(0,0,0,0.12) !important;
    color: var(--chalk-faint) !important;
    border-color: var(--chalk-faint) !important;
  }
  .op-btn-danger {
    color: var(--brand);
    border-color: var(--brand);
  }
  .ep-root :global(.ant-btn).op-btn-danger {
    color: var(--brand) !important;
    border-color: var(--brand) !important;
  }
  .op-btn-danger:hover { background: rgba(200, 57, 43, 0.12); }
  .ep-root :global(.ant-btn).op-btn-danger:hover {
    background: rgba(200, 57, 43, 0.12) !important;
  }
  .op-btn-start {
    min-width: 200px;
    background: var(--chalk-bright);
    border-color: var(--ink);
    color: var(--ink);
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.25);
  }
  .ep-root :global(.ant-btn).op-btn-start {
    background: var(--chalk-bright) !important;
    border-color: var(--ink) !important;
    color: var(--ink) !important;
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.25) !important;
  }
  .op-btn-start:hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.25);
  }
  .op-btn-start:active { transform: translate(0, 0); box-shadow: none; }
  .ep-root :global(.ant-btn).op-btn-start:active { box-shadow: none !important; }
  .op-btn-start:disabled {
    background: rgba(0,0,0,0.12);
    border-color: var(--chalk-faint);
    color: var(--chalk-faint);
    box-shadow: none;
  }
  .ep-root :global(.ant-btn).op-btn-start:disabled {
    background: rgba(0,0,0,0.12) !important;
    border-color: var(--chalk-faint) !important;
    color: var(--chalk-faint) !important;
    box-shadow: none !important;
  }

  /* ===== 评分反馈结构化卡片 ===== */
  .ep-eval-section {
    margin-bottom: 14px;
  }
  .ep-eval-section:last-child { margin-bottom: 0; }
  .ep-eval-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--chalk-yellow);
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .ep-eval-body {
    font-size: 14px;
    line-height: 1.7;
    color: var(--chalk);
    padding-left: 10px;
    border-left: 2px solid var(--chalk-dim);
  }
  .ep-eval-body p { color: var(--chalk); }
  .ep-eval-body strong { color: var(--chalk-bright); }

  /* ===== 要点命中条目 ===== */
  .op-point {
    padding: 8px 10px;
    border: 1.5px solid var(--chalk-dim);
    border-radius: 3px;
    font-size: 12px;
    background: var(--board-deep);
  }

  /* 进度条 → 见 globals.css .ep-root 段 */

  /* Alert / Divider / Progress / Markdown 元素覆盖 → 见 globals.css .ep-root 段 */

  @media (prefers-reduced-motion: reduce) {
    .op-card, .op-input, .op-btn, .op-textarea {
      transition: opacity 100ms ease;
    }
  }
`;

export default function ExaminerPanel({ sessionId, collectionName }: Props) {
  const [phase, setPhase] = useState<'config' | 'exam'>('config');
  const [state, setState] = useState<ExamState | null>(null);
  const [answer, setAnswer] = useState('');
  // 最近一次成功提交的答案，评分解析失败时供"重试"原样重发
  const [lastSubmittedAnswer, setLastSubmittedAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [targetPosition, setTargetPosition] = useState('');
  const [topic, setTopic] = useState('');
  const [chalkStage, setChalkStage] = useState<'idle' | 'start'>('idle'); // 粉笔盒起跳动画（仅配置页）

  const fetchSessionConfig = useCallback(async () => {
    try {
      const res = await api.get(`/api/sessions/${sessionId}`, { skipErrorToast: true });
      const examState = res.data?.exam_state || {};
      if (examState.target_position && examState.topic) {
        setTargetPosition(examState.target_position);
        setTopic(examState.topic);
      }
    } catch {
      // ignore
    }
  }, [sessionId]);

  const fetchState = useCallback(async () => {
    try {
      const res = await api.get(`/api/exam/${sessionId}`, { skipErrorToast: true });
      if (res.data?.status && res.data.status !== 'configuring') {
        setState(res.data);
        setPhase('exam');
      }
    } catch {
      // 未开始面试则停留在配置页
    }
  }, [sessionId]);

  useEffect(() => {
    fetchSessionConfig();
    fetchState();
    setLastSubmittedAnswer('');
  }, [fetchSessionConfig, fetchState]);

  const startExam = async () => {
    if (!targetPosition.trim() || !topic.trim()) return;
    setLoading(true);
    setChalkStage('start');
    try {
      const res = await api.post('/api/exam/start', {
        session_id: sessionId,
        target_position: targetPosition.trim(),
        topic: topic.trim(),
        collection_name: collectionName,
        top_k: 5,
      });
      setState(res.data);
      setPhase('exam');
      setAnswer('');
      setLastSubmittedAnswer('');
    } catch {
      // 拦截器已统一弹出错误提示
      setChalkStage('idle');
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async (value?: string) => {
    const text = (value ?? answer).trim();
    if (!text || !state) return;
    setLoading(true);
    try {
      const res = await api.post(`/api/exam/${state.session_id}/next`, {
        answer: text,
        top_k: 5,
      });
      setState(res.data);
      setLastSubmittedAnswer(text);
      setAnswer('');
    } catch {
      // 拦截器已统一弹出错误提示
    } finally {
      setLoading(false);
    }
  };

  // 放弃本题 / 结束面试：走后端 action 字段，不再用魔法字符串充当答案
  const submitAction = async (action: 'give_up' | 'end') => {
    if (!state || loading) return;
    setLoading(true);
    try {
      const res = await api.post(`/api/exam/${state.session_id}/next`, {
        action,
        answer: '',
        top_k: 5,
      });
      setState(res.data);
      setAnswer('');
    } catch {
      // 拦截器已统一弹出错误提示
    } finally {
      setLoading(false);
    }
  };

  const avgScore = state?.scores?.length
    ? state.scores.reduce((s, i) => s + i.score, 0) / state.scores.length
    : 0;

  const progressPercent = Math.min(((state?.scores?.length || 0) / MAX_QUESTIONS) * 100, 100);

  if (phase === 'config') {
    return (
      <div className="ep-wall">
        <LeftBulletinBoard />
        <div className="ep-root" style={{ padding: '40px 24px', minHeight: '100%' }}>
          <CornerStar /><CornerLamp />
          <ChalkBox wholeCount={MAX_QUESTIONS} stubCount={0} animStage={chalkStage} />
        <div className="op-card">
          <div className="op-card-header ep-config-header">
            <span>配置模拟面试</span>
            <div className="ep-config-sub">考官已就位，先填考点</div>
          </div>
          <div style={{ padding: 24 }}>
            <Space direction="vertical" size={20} style={{ width: '100%' }}>
              <div>
                <Text style={{ display: 'block', marginBottom: 6, fontSize: 12, fontWeight: 600, letterSpacing: 1, color: 'var(--chalk-yellow)' }}>目标岗位</Text>
                <Input
                  className="op-input"
                  placeholder="例如：后端开发工程师"
                  value={targetPosition}
                  onChange={(e) => setTargetPosition(e.target.value)}
                  size="large"
                />
              </div>
              <div>
                <Text style={{ display: 'block', marginBottom: 6, fontSize: 12, fontWeight: 600, letterSpacing: 1, color: 'var(--chalk-yellow)' }}>面试方向</Text>
                <Input
                  className="op-input"
                  placeholder="例如：Java 并发 / Redis / Vue 响应式"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  size="large"
                />
              </div>
              <div>
                <span className="ep-chalk-note">本场 5 题 · 单题最多追问 2 次</span>
              </div>
              <div style={{ textAlign: 'center' }}>
                <Button
                  className="op-btn op-btn-start"
                  size="large"
                  icon={<Play size={16} />}
                  onClick={startExam}
                  loading={loading}
                  disabled={!targetPosition.trim() || !topic.trim()}
                >
                  开始面试
                </Button>
              </div>
            </Space>
          </div>
        </div>

        {/* 常驻板书 · 考场规则（一-6 落地） */}
        <div className="ep-board-rules" style={{ marginTop: 24, textAlign: 'center' }}>
          <div className="ep-board-rules-text" style={{ fontFamily: 'var(--font-display)', fontSize: 13, color: 'var(--chalk-dim)', lineHeight: 2.2, letterSpacing: 1 }}>
            一、独立作答，诚信为本<br />
            二、每题限时思考，先理解再落笔<br />
            三、结束面试后查看成绩单
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--chalk-faint)', fontFamily: 'var(--font-display)', letterSpacing: 2 }}>
            —— 值日生：DX
          </div>
        </div>
        <style jsx>{examinerStyles}</style>
      </div>
        <RightBulletinBoard />
      </div>
    );
  }

  if (!state) return <Spin style={{ margin: '40px auto', display: 'block' }} />;

  return (
    <div className="ep-wall">
      <LeftBulletinBoard isExam />
      <div className="ep-root" style={{ padding: '24px', minHeight: '100%' }}>
      <CornerStar /><CornerLamp />
      <ChalkBox
        wholeCount={Math.max(0, MAX_QUESTIONS - (state?.scores?.length || 0))}
        stubCount={Math.min(state?.scores?.length || 0, MAX_QUESTIONS)}
        animStage={state?.evaluation ? 'score' : 'idle'}
      />
      <div className="op-card op-card-lift" style={{ marginBottom: 16, padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <Space size={16}>
            <Target size={24} color="var(--chalk-yellow)" />
            <div>
              <Title level={5} style={{ margin: 0, fontSize: 16, color: 'var(--chalk-bright)', fontFamily: 'var(--font-display)' }}>
                模拟面试 · {state.question_index} / {MAX_QUESTIONS} 题
              </Title>
              <Text type="secondary" style={{ fontSize: 12, color: 'var(--chalk)' }}>
                {state.status === 'follow_up' ? '追问环节' : '正式题目'}
                {state.follow_up_count > 0 && ` · 已追问 ${state.follow_up_count} 次`}
              </Text>
            </div>
          </Space>
          <Space size={24} align="center">
            <div style={{ textAlign: 'right' }}>
              <Text type="secondary" style={{ fontSize: 12, display: 'block', color: 'var(--chalk-dim)' }}>当前均分</Text>
              <Text strong style={{ fontSize: 24, fontFamily: 'var(--font-pixel)', color: 'var(--chalk-yellow)', lineHeight: 1.2 }}>
                {avgScore.toFixed(1)}
              </Text>
              <Text type="secondary" style={{ fontSize: 12, color: 'var(--chalk-dim)' }}> / 10</Text>
            </div>
            <div style={{ width: 160 }}>
              <Progress
                className="ep-progress"
                percent={progressPercent}
                size="small"
                strokeColor="var(--brand)"
                trailColor="var(--chalk-weak)"
              />
            </div>
          </Space>
        </div>
      </div>

      <div className="op-card op-card-lift" style={{ marginBottom: 16, padding: 20 }}>
        <div className="ep-question">
          <ReactMarkdown>{state.current_question}</ReactMarkdown>
        </div>
        {state.current_expectations && state.current_expectations.length > 0 && (
          <div className="ep-expectation-wrap">
            {state.current_expectations.slice(0, 8).map((e, i) => (
              <span key={i} className="ep-expectation-tag" title={`考察：${e}`}>
                考察：{e}
              </span>
            ))}
            {state.current_expectations.length > 8 && (
              <span className="op-tag-sunken">+{state.current_expectations.length - 8} 更多</span>
            )}
          </div>
        )}
      </div>

      {state.reference_points && state.reference_points.length > 0 && (
        <ReferencePointsCard points={state.reference_points} />
      )}

      {state.status !== 'finished' && (
        <div className="op-card op-card-lift" style={{ marginBottom: 16, padding: 20 }}>
          <TextareaAutosize
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => {
              if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                submitAnswer();
              }
            }}
            placeholder="请组织语言回答，不要直接复制参考资料…"
            minRows={3}
            maxRows={8}
            disabled={loading}
            className="op-textarea"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 }}>
            <Space>
              <Button className="op-btn" onClick={() => submitAction('give_up')} disabled={loading}>
                不知道
              </Button>
              <Button className="op-btn op-btn-danger" onClick={() => submitAction('end')} disabled={loading}>
                结束面试
              </Button>
            </Space>
            <Button
              className="op-btn op-btn-primary"
              icon={<Send size={16} />}
              onClick={() => submitAnswer()}
              loading={loading}
              disabled={!answer.trim()}
            >
              提交答案
            </Button>
          </div>
        </div>
      )}

      {state.evaluation && (
        <div className="op-card op-card-lift" style={{ marginBottom: 16 }}>
          <div className="op-card-header">
            <Space>
              {state.cheating_detected ? (
                <AlertTriangle size={18} color="var(--cite-4)" />
              ) : (
                <CheckCircle2 size={18} color={scoreColor(state.evaluation.score)} />
              )}
              <span>评分反馈</span>
              <span
                className="op-tag"
                style={{
                  color: 'var(--chalk-yellow)',
                  fontFamily: 'var(--font-pixel)',
                  fontSize: 12,
                }}
              >
                {state.evaluation.score}<span style={{ fontFamily: 'inherit', color: 'var(--chalk-dim)', fontSize: 11 }}> / 10 分</span>
              </span>
            </Space>
          </div>
          <div style={{ padding: 16 }}>
            {state.cheating_detected && (
              <Alert
                className="ep-alert"
                message="反作弊提示"
                description="系统检测到您的回答与参考资料或历史回答高度重合。模拟面试要求独立作答。"
                type="warning"
                showIcon
                style={{
                  marginBottom: 12,
                  background: 'rgba(200,57,43,0.12)',
                  border: '1.5px solid var(--chalk-bright)',
                  borderRadius: 3,
                  boxShadow: 'none',
                }}
              />
            )}
            <EvaluationCard
              evaluation={state.evaluation}
              onRetry={lastSubmittedAnswer ? () => submitAnswer(lastSubmittedAnswer) : undefined}
            />
          </div>
        </div>
      )}

      {state.summary && (
        <div className="op-card op-card-lift">
          <div className="op-card-header">
            <Space>
              <Flag size={18} color="var(--chalk-yellow)" />
              <span>面试总结</span>
              <span className="op-tag" style={{ color: 'var(--chalk-yellow)', fontFamily: 'var(--font-pixel)', fontSize: 12 }}>
                总体 {state.summary.total_score}<span style={{ fontFamily: 'inherit', color: 'var(--chalk-dim)', fontSize: 11 }}> / 100 分</span>
              </span>
            </Space>
          </div>
          <div style={{ padding: 16 }}>
            {/* 励志粉笔大字 */}
            <div className="ep-motto" style={{
              fontFamily: 'var(--font-display)', fontSize: 24, color: 'var(--chalk)',
              textAlign: 'center', marginBottom: 18, opacity: 0.75, letterSpacing: 2,
            }}>
              天道酬勤
            </div>
            {/* 成绩章预留位 */}
            <div className="ep-stamp-placeholder" style={{
              width: 72, height: 72, borderRadius: '50%', margin: '0 auto 16px',
              border: '1.5px dashed var(--chalk-faint)', display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              color: 'var(--chalk-faint)', fontSize: 11, fontFamily: 'var(--font-display)',
            }}>
              成绩章
            </div>
            <div className="markdown-body" style={{ fontSize: 14, color: 'var(--chalk)' }}>
              <ReactMarkdown>{state.summary.raw}</ReactMarkdown>
            </div>
            {state.weak_points && state.weak_points.length > 0 && (
              <>
                <Divider style={{ borderColor: 'var(--chalk-weak)' }} />
                <div>
                  <Text strong style={{ fontSize: 13, color: 'var(--chalk-yellow)' }}>
                    高频遗漏点
                  </Text>
                  <ul style={{ marginTop: 8, paddingLeft: 18, color: 'var(--chalk)', fontSize: 13, lineHeight: 1.7 }}>
                    {state.weak_points.slice(0, 10).map((p, i) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
            <Divider style={{ borderColor: 'rgba(240,237,228,0.2)' }} />
            <Button
              className="op-btn"
              icon={<RotateCcw size={16} />}
              onClick={() => {
                setPhase('config');
                setState(null);
                setAnswer('');
                setLastSubmittedAnswer('');
              }}
            >
              重新开始
            </Button>
          </div>
        </div>
      )}

      <style jsx>{examinerStyles}</style>
    </div>
      <RightBulletinBoard isExam />
    </div>
  );
}
