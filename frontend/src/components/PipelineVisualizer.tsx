import React from 'react';
import type { PipelineStep } from '../types';
import { CheckIcon, BrainIcon } from './Icons';

interface PipelineVisualizerProps {
  steps: PipelineStep[];
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ steps }) => {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="bg-slate-950/70 border border-slate-800/90 rounded-2xl p-4 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <BrainIcon className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Agentic Orchestration Pipeline ({steps.length} Specialist Tasks)
          </span>
        </div>
        <span className="text-[11px] text-emerald-400 font-mono font-medium">
          ✓ Verified Deterministic
        </span>
      </div>

      <div className="space-y-2 relative">
        {steps.map((step, idx) => {
          const isDone = step.status === 'completed';
          return (
            <div
              key={step.step_id}
              className={`p-2.5 rounded-xl border transition-all text-xs ${
                isDone
                  ? 'bg-slate-900/80 border-slate-800 text-slate-200'
                  : 'bg-slate-950 border-slate-800/40 text-slate-500'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                      isDone
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {isDone ? <CheckIcon className="w-3 h-3" /> : idx + 1}
                  </div>
                  <span className="font-semibold text-slate-200">{step.step_name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-cyan-400 border border-slate-700">
                    {step.tool_name}
                  </span>
                  {step.duration_ms > 0 && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      {step.duration_ms}ms
                    </span>
                  )}
                </div>
              </div>

              {step.evidence_generated && (
                <div className="mt-1.5 pl-7 text-[11px] text-slate-400 flex items-center space-x-1">
                  <span className="text-cyan-400">↳ Evidence:</span>
                  <span>{step.evidence_generated}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
