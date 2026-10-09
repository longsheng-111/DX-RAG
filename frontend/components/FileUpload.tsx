'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Upload, Select, message, Typography, Space, Progress, Alert,
} from 'antd';
import { InboxOutlined, CloudUploadOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import axios from 'axios';

const { Dragger } = Upload;
const { Text } = Typography;

const CheckmarkSvg = () => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M4 13l5 5 11-11" />
  </svg>
);

interface Props { collectionName: string; }

interface Collection { name: string; chunk_count: number; }

export default function FileUpload({ collectionName }: Props) {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [targetCollection, setTargetCollection] = useState(collectionName);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState<{
    file_name: string; chunks: number; collection_name: string;
  } | null>(null);

  const fetchCollections = useCallback(async () => {
    try {
      const res = await axios.get('/api/collections');
      setCollections(res.data.collections || []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { fetchCollections(); }, [fetchCollections]);
  useEffect(() => { setTargetCollection(collectionName); }, [collectionName]);

  // 阶段 4：上传接口改为异步队列——提交后拿到 job_id 立即返回，
  // 这里按 2s 间隔轮询任务状态直到 done / failed / cancelled
  const pollJob = useCallback((jobId: string): Promise<any> => {
    const startedAt = Date.now();
    return new Promise((resolve, reject) => {
      const timer = setInterval(async () => {
        try {
          const res = await axios.get(`/api/ingest-jobs/${jobId}`);
          const job = res.data;
          setProgress(Math.round((job.progress || 0) * 100));
          if (job.status === 'done') {
            clearInterval(timer);
            resolve(job);
          } else if (job.status === 'failed' || job.status === 'cancelled') {
            clearInterval(timer);
            reject(new Error(job.error || '摄入任务失败'));
          } else if (Date.now() - startedAt > 30 * 60 * 1000) {
            clearInterval(timer);
            reject(new Error('摄入超时（30 分钟），请稍后在文件管理中确认结果'));
          }
        } catch (err: any) {
          clearInterval(timer);
          reject(err);
        }
      }, 2000);
    });
  }, []);

  const uploadProps: UploadProps = {
    name: 'file', multiple: false, showUploadList: false,
    accept: '.txt,.md,.csv,.json,.log,.pdf,.docx,.xlsx,.xlsm,.xltx,.xltm',
    beforeUpload: (file) => {
      if (file.size > 50 * 1024 * 1024) { message.error('文件过大（最大 50MB）'); return false; }
      return true;
    },
    customRequest: async ({ file, onSuccess, onError }) => {
      setUploading(true); setUploadResult(null); setProgress(0);
      const formData = new FormData();
      formData.append('file', file as File);
      formData.append('collection_name', targetCollection);
      try {
        const res = await axios.post('/api/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        // 上传已落盘，摄入在后台队列执行；轮询等待最终结果
        const job = await pollJob(res.data.job_id);
        const result = {
          file_name: job.file_name,
          chunks: job.chunk_count ?? 0,
          collection_name: job.collection_name,
        };
        setUploadResult(result);
        message.success(`上传完成！共 ${result.chunks} 个切片`);
        onSuccess?.(result as any, undefined as any);
        fetchCollections();
      } catch (err: any) {
        message.error(err.response?.data?.detail || err.message || '上传失败');
        onError?.(err as any);
      } finally { setUploading(false); }
    },
  };

  const collectionOptions = collections.map((c) => ({
    value: c.name, label: `${c.name} (${c.chunk_count} chunks)`,
  }));
  if (!collectionOptions.find((o) => o.value === targetCollection)) {
    collectionOptions.unshift({ value: targetCollection, label: targetCollection });
  }

  const formatTypes = [
    { type: '文档', tags: ['.pdf', '.docx', '.md', '.txt'] },
    { type: '数据', tags: ['.xlsx', '.csv', '.json'] },
  ];

  return (
    <div className="fu-root">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <CloudUploadOutlined style={{ fontSize: 22, color: 'var(--brand)' }} />
          <h2 style={{ margin: 0, color: 'var(--ink)', fontFamily: 'var(--font-display)' }}>上传文件</h2>
        </div>
      </div>

      <div className="op-card" style={{ marginBottom: 20, padding: 24 }}>
        <div className="op-bar">
          <Text strong style={{ color: 'var(--ink)', fontSize: 14 }}>
            目标知识库
          </Text>
          <Select
            value={targetCollection}
            onChange={setTargetCollection}
            options={collectionOptions}
            style={{ minWidth: 240, flex: 1 }}
            className="op-select"
          />
        </div>

        <Dragger {...uploadProps} disabled={uploading} className="op-upload">
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ color: 'var(--brand)', fontSize: 48 }} />
          </p>
          <p className="ant-upload-text" style={{ fontSize: 17, fontWeight: 600, color: 'var(--ink)' }}>
            点击或拖拽文件到此处
          </p>
          <p className="ant-upload-hint" style={{ fontSize: 13, color: 'var(--ink-secondary)', maxWidth: 420, margin: '8px auto 0' }}>
            支持 PDF、Word、Excel、Markdown、TXT — 单个文件最大 50MB
          </p>
        </Dragger>

        {uploading && (
          <div style={{ marginTop: 20 }}>
            <Progress
              className="op-progress"
              percent={progress}
              status="active"
              format={(p) => (p && p >= 95 ? '即将完成...' : `摄入中 ${p ?? 0}%`)}
              strokeColor="var(--brand)"
              trailColor="var(--bg-sunken)"
              strokeLinecap="butt"
              steps={12}
            />
          </div>
        )}
      </div>

      {uploadResult && (
        <Alert
          type="success"
          showIcon
          icon={<span className="op-success-icon"><CheckmarkSvg /></span>}
          message="上传成功"
          description={
            <Space direction="vertical" size={4}>
              <Text><strong>文件：</strong> {uploadResult.file_name}</Text>
              <Text><strong>知识库：</strong> {uploadResult.collection_name}</Text>
              <Text><strong>切片数：</strong> {uploadResult.chunks.toLocaleString()}</Text>
            </Space>
          }
          className="op-alert-success"
          style={{
            marginBottom: 20,
            background: 'var(--bg-paper)',
            border: '1.5px solid var(--ink)',
            borderRadius: 'var(--radius)',
            boxShadow: 'none',
          }}
        />
      )}

      <div className="op-card">
        <div className="op-card-header">支持格式</div>
        <div style={{ padding: '20px 24px' }}>
          <div style={{ display: 'grid', gap: 16 }}>
            {formatTypes.map((group) => (
              <div
                key={group.type}
                className="op-format-group"
              >
                <Text strong style={{ fontSize: 13, color: 'var(--ink)', display: 'block', marginBottom: 10 }}>
                  {group.type}
                </Text>
                <Space wrap size={8}>
                  {group.tags.map((t) => (
                    <span key={t} className="op-tag-sunken" style={{ fontFamily: '"JetBrains Mono", "SF Mono", Consolas, monospace' }}>
                      {t}
                    </span>
                  ))}
                </Space>
              </div>
            ))}
          </div>
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 16, color: 'var(--ink-faint)' }}>
            * 扫描版 PDF 需要配置 DashScope API Key 进行 OCR 识别
          </Text>
        </div>
      </div>

      <style jsx>{`
        .fu-root {
          color: var(--ink);
        }
        .op-progress :global(.ant-progress-steps-item) {
          border-radius: 0 !important;
        }
        .op-success-icon {
          color: var(--brand);
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }
        .op-alert-success :global(.ant-alert-icon) {
          color: var(--brand);
        }
      `}</style>
    </div>
  );
}
