'use client';

import { useState } from 'react';
import { FileText, Search, MessageSquare, Upload } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { chatRequest, chatWithPdf } from '@/lib/api';

export default function QuerifyHome() {
  const [message, setMessage] = useState('');
  const [reply, setReply] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfMeta, setPdfMeta] = useState<Record<string, any> | null>(null);

  /* ---------------- Chat Handlers ---------------- */

  async function handleChat() {
    if (!message.trim() || loading) return;

    setLoading(true);
    setError(null);
    setPdfMeta(null);

    try {
      const data = await chatRequest(
        message,
        '🤖 Smart Mode (Auto-detect)'
      );
      setReply(data.reply);
      setMessage('');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handlePdfChat() {
    if (!pdfFile || loading) return;

    setLoading(true);
    setError(null);
    setReply('');

    try {
      const data = await chatWithPdf(pdfFile, message);
      setReply(data.reply);
      setPdfMeta(data.metadata);
      setMessage('');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 text-slate-100">

      {/* Sidebar */}
      <aside className="w-64 px-6 py-8 flex flex-col bg-black/40 backdrop-blur-lg border-r border-white/10">
        <h1 className="text-2xl font-bold mb-8 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          Querify
        </h1>

        <Input
          placeholder="Search..."
          className="mb-8 bg-white/5 border-white/10 text-slate-200 placeholder:text-slate-400 rounded-lg focus:ring-2 focus:ring-indigo-500"
        />

        <nav className="space-y-6 text-sm text-slate-300">
          <SidebarItem icon={<Search size={18} />} label="Explore" />
          <SidebarItem icon={<MessageSquare size={18} />} label="Library" />
          <SidebarItem icon={<FileText size={18} />} label="Files" />
        </nav>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col items-center px-10 pt-20 relative overflow-y-auto">
        <h2 className="text-5xl font-bold bg-gradient-to-r from-indigo-300 to-purple-300 bg-clip-text text-transparent">
          Welcome to Querify
        </h2>
        <p className="text-slate-400 mt-3 mb-12 text-lg text-center max-w-xl">
          Upload PDFs, summarize documents, and ask questions instantly using AI.
        </p>

        {/* Action Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 mb-12 w-full max-w-5xl">
          <ActionCard
            title="Ask from PDF"
            description="Upload a PDF and ask AI any question about it"
            color="from-indigo-500 to-indigo-700"
          >
            <label className="flex items-center gap-2 text-sm cursor-pointer mt-3">
              <Upload size={16} />
              <span>Upload PDF</span>
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(e) =>
                  setPdfFile(e.target.files?.[0] || null)
                }
              />
            </label>
            {pdfFile && (
              <p className="mt-2 text-xs text-slate-200">
                {pdfFile.name}
              </p>
            )}
          </ActionCard>

          <ActionCard
            title="Summarize Document"
            description="Leave the question empty to get a summary"
            color="from-purple-500 to-purple-700"
          />

          <ActionCard
            title="Deep Research"
            description="Analyze long documents and extract insights"
            color="from-teal-500 to-teal-700"
          />
        </div>

        {/* Reply */}
        {reply && (
          <div className="max-w-3xl w-full mb-24 p-6 bg-black/40 border border-white/10 rounded-xl whitespace-pre-wrap">
            {reply}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-24 text-red-400">
            Error: {error}
          </div>
        )}

        {/* Chat Input */}
        <div className="w-full max-w-3xl fixed bottom-8 px-6">
          <div className="flex gap-3">
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  pdfFile ? handlePdfChat() : handleChat();
                }
              }}
              placeholder={
                pdfFile
                  ? 'Ask something about your PDF (or press Enter for summary)'
                  : 'Ask something...'
              }
              className="rounded-full px-6 py-4 shadow-2xl bg-black/50 border-white/15 text-slate-200 placeholder:text-slate-400 backdrop-blur-xl focus:ring-2 focus:ring-indigo-500"
            />

            <Button
              onClick={pdfFile ? handlePdfChat : handleChat}
              disabled={loading}
              className="rounded-full px-6"
            >
              {loading ? 'Thinking…' : 'Send'}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}

/* ---------------- Components ---------------- */

function SidebarItem({
  icon,
  label,
}: {
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <div className="flex items-center gap-3 cursor-pointer hover:text-white transition-colors duration-200">
      {icon}
      <span>{label}</span>
    </div>
  );
}

function ActionCard({
  title,
  description,
  color,
  children,
}: {
  title: string;
  description: string;
  color: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={`p-6 rounded-2xl bg-gradient-to-br ${color} bg-opacity-20 border border-white/10 hover:shadow-2xl hover:scale-105 transform transition-all duration-300`}
    >
      <h3 className="font-semibold text-lg text-white">{title}</h3>
      <p className="text-sm text-slate-300 mt-2">{description}</p>
      {children}
    </div>
  );
}
