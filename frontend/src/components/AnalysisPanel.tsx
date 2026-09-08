import React from 'react';
import { AlertIcon, BrainIcon, SparklesIcon, CheckIcon } from './Icons';

interface AnalysisPanelProps {
  query: string;
  onQueryChange: (q: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  presetQueries: string[];
  answer: string | null;
  toolsUsed: string[];
  confidence: number | null;
  limitations: string | null;
  executionTimeMs: number | null;
}

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  query,
  onQueryChange,
  onSubmit,
  isLoading,
  presetQueries,
  answer,
  toolsUsed,
  confidence,
  limitations,
  executionTimeMs,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="w-96 border-l border-slate-800 bg-slate-900/60 backdrop-blur flex flex-col justify-between shrink-0 h-[calc(100vh-65px)] overflow-hidden">
      {/* Scrollable Content */}
      <div className="p-4 space-y-4 overflow-y-auto flex-1">
        {/* Preset Query Chips */}
        <div>
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <SparklesIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Recommended Queries</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {presetQueries.map((pq, idx) => (
              <button
                key={idx}
                onClick={() => onQueryChange(pq)}
                className="text-left text-xs bg-slate-950/80 hover:bg-slate-800 text-slate-300 hover:text-white px-2.5 py-1.5 rounded-xl border border-slate-800 transition shadow-sm leading-relaxed"
              >
                {pq}
              </button>
            ))}
          </div>
        </div>

        {/* Answer Display */}
        {answer && (
          <div className="space-y-3">
            {/* Header info bar */}
            <div className="flex items-center justify-between bg-slate-950/80 px-3 py-2 rounded-xl border border-slate-800">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Analyst Assessment
                </span>
              </div>
              <div className="flex items-center space-x-2 text-xs font-mono">
                {confidence !== null && (
                  <span className="text-emerald-400 font-bold">
                    {Math.round(confidence * 100)}% Conf
                  </span>
                )}
                {executionTimeMs !== null && (
                  <span className="text-slate-500">
                    {executionTimeMs}ms
                  </span>
                )}
              </div>
            </div>

            {/* Answer Content */}
            <div className="bg-slate-950/70 border border-slate-800/90 rounded-2xl p-4 shadow-xl text-xs text-slate-300 space-y-2 leading-relaxed">
              {answer.split('\n\n').map((paragraph, pIdx) => {
                if (paragraph.startsWith('### ')) {
                  return (
                    <h3 key={pIdx} className="text-sm font-bold text-cyan-300 mt-2 mb-1">
                      {paragraph.replace('### ', '')}
                    </h3>
                  );
                }
                if (paragraph.startsWith('> ⚠️')) {
                  return null; // Handled separately in limitations card
                }
                return (
                  <div key={pIdx} className="space-y-1">
                    {paragraph.split('\n').map((line, lIdx) => {
                      if (line.startsWith('- ')) {
                        return (
                          <div key={lIdx} className="flex items-start space-x-2 pl-2">
                            <span className="text-cyan-400 mt-1">•</span>
                            <span dangerouslySetInnerHTML={{ __html: formatMarkdown(line.substring(2)) }} />
                          </div>
                        );
                      }
                      return (
                        <p key={lIdx} dangerouslySetInnerHTML={{ __html: formatMarkdown(line) }} />
                      );
                    })}
                  </div>
                );
              })}
            </div>

            {/* Tools Used Badges */}
            {toolsUsed.length > 0 && (
              <div>
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">
                  Specialist Tools Invoked:
                </div>
                <div className="flex flex-wrap gap-1">
                  {toolsUsed.map((tool, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded-lg bg-cyan-950/50 border border-cyan-800/60 text-cyan-400 text-[10px] font-mono flex items-center space-x-1"
                    >
                      <CheckIcon className="w-2.5 h-2.5" />
                      <span>{tool}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Limitations Notice */}
            {limitations && (
              <div className="bg-amber-950/30 border border-amber-800/60 rounded-xl p-3 text-xs text-amber-300 space-y-1">
                <div className="flex items-center space-x-1.5 font-bold text-[11px] uppercase tracking-wider text-amber-400">
                  <AlertIcon className="w-3.5 h-3.5" />
                  <span>Physical & Sensor Constraints</span>
                </div>
                <p className="text-[11px] text-amber-200/90 leading-normal">
                  {limitations}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Loading Spinner State */}
        {isLoading && (
          <div className="py-12 flex flex-col items-center justify-center space-y-3 text-slate-400">
            <div className="w-8 h-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
            <div className="text-xs font-semibold tracking-wide text-cyan-400">
              Executing Agentic Specialist Pipeline...
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              Extracting bands • Calculating indices • Grounding features
            </div>
          </div>
        )}
      </div>

      {/* Query Input Box */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/90">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a natural-language query (e.g. 'Where is vegetation concentrated?', 'What changed?')..."
            rows={3}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-xl p-3 pr-10 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition resize-none shadow-inner"
          />
          <button
            onClick={onSubmit}
            disabled={isLoading || !query.trim()}
            className="absolute bottom-3 right-3 p-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 disabled:text-slate-600 text-slate-950 font-bold transition shadow-md shadow-cyan-500/20"
            title="Execute Query"
          >
            <BrainIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="text-[10px] text-slate-500 mt-1.5 flex items-center justify-between">
          <span>Press Enter ↵ to dispatch query</span>
          <span className="font-mono text-cyan-400/80">SIH 2026 Ready</span>
        </div>
      </div>
    </div>
  );
};

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
    .replace(/\[(.*?)\]/g, '<span class="text-cyan-400 font-mono">$1</span>');
}
