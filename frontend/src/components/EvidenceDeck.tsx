import React from 'react';
import type { BoundingBox, VisualEvidence } from '../types';
import { DownloadIcon, CrosshairIcon } from './Icons';

interface EvidenceDeckProps {
  boundingBoxes: BoundingBox[];
  evidenceLayers: VisualEvidence[];
  metrics: Record<string, any>;
  hoveredBoxId?: string | null;
  onHoverBox?: (boxId: string | null) => void;
  onExportReport: () => void;
}

export const EvidenceDeck: React.FC<EvidenceDeckProps> = ({
  boundingBoxes,
  metrics,
  hoveredBoxId,
  onHoverBox,
  onExportReport,
}) => {
  return (
    <div className="bg-slate-950/70 border border-slate-800/90 rounded-2xl p-4 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <CrosshairIcon className="w-4 h-4 text-cyan-400" /> Quantitative Evidence Deck
        </span>
        <button
          onClick={onExportReport}
          className="flex items-center space-x-1.5 px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-xl text-xs font-semibold border border-slate-700 transition shadow-sm"
        >
          <DownloadIcon className="w-3.5 h-3.5" />
          <span>Export Dossier</span>
        </button>
      </div>

      {/* Metrics Chips */}
      {Object.keys(metrics).length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(metrics).map(([key, val]) => {
            if (typeof val === 'object' || Array.isArray(val)) return null;
            const label = key.replace(/_/g, ' ');
            return (
              <div key={key} className="bg-slate-900/90 border border-slate-800 p-2.5 rounded-xl">
                <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  {label}
                </div>
                <div className="text-sm font-bold text-white font-mono mt-0.5">
                  {typeof val === 'number' ? val.toLocaleString() : String(val)}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Spatial Clusters List */}
      {boundingBoxes.length > 0 && (
        <div>
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">
            Identified Spatial Clusters ({boundingBoxes.length})
          </div>
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {boundingBoxes.map((b) => {
              const isHovered = hoveredBoxId === b.id;
              return (
                <div
                  key={b.id}
                  onMouseEnter={() => onHoverBox?.(b.id)}
                  onMouseLeave={() => onHoverBox?.(null)}
                  className={`p-2 rounded-xl border text-xs cursor-pointer transition flex items-center justify-between ${
                    isHovered
                      ? 'bg-amber-950/40 border-amber-500/60 text-white'
                      : 'bg-slate-900/60 border-slate-800/80 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    <span className="font-semibold">{b.label}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {b.area_hectares && (
                      <span className="text-[10px] text-slate-400 font-mono">
                        {b.area_hectares} ha
                      </span>
                    )}
                    {b.confidence && (
                      <span className="text-[10px] font-bold text-emerald-400 font-mono px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/60">
                        {Math.round(b.confidence * 100)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
