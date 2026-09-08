import React from 'react';
import { SatelliteIcon, SplitIcon, RadarIcon, LayersIcon, UploadIcon } from './Icons';
import type { SampleDataset } from '../types';

interface SidebarProps {
  activeMode: 'single' | 'change' | 'sar';
  onModeChange: (mode: 'single' | 'change' | 'sar') => void;
  samples: SampleDataset[];
  selectedSampleId: string;
  onSelectSample: (sample: SampleDataset) => void;
  onUploadClick: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeMode,
  onModeChange,
  samples,
  selectedSampleId,
  onSelectSample,
  onUploadClick,
}) => {
  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/60 backdrop-blur flex flex-col justify-between shrink-0 h-[calc(100vh-65px)] overflow-y-auto">
      <div className="p-4 space-y-6">
        {/* Modes */}
        <div>
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 px-2">
            Analysis Modalities
          </div>
          <nav className="space-y-1">
            <button
              onClick={() => onModeChange('single')}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                activeMode === 'single'
                  ? 'bg-cyan-600/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <SatelliteIcon className="w-4 h-4 text-cyan-400" />
              <span>Single Image VQA & Spectral</span>
            </button>

            <button
              onClick={() => onModeChange('change')}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                activeMode === 'change'
                  ? 'bg-amber-600/20 text-amber-300 border border-amber-500/40'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <SplitIcon className="w-4 h-4 text-amber-400" />
              <span>Change Detection (T1 vs T2)</span>
            </button>

            <button
              onClick={() => onModeChange('sar')}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                activeMode === 'sar'
                  ? 'bg-purple-600/20 text-purple-300 border border-purple-500/40'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <RadarIcon className="w-4 h-4 text-purple-400" />
              <span>Optical + SAR Fusion</span>
            </button>
          </nav>
        </div>

        {/* Upload Button */}
        <div>
          <button
            onClick={onUploadClick}
            className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-sm font-semibold transition shadow-sm hover:border-slate-600"
          >
            <UploadIcon className="w-4 h-4 text-cyan-400" />
            <span>Upload Satellite Raster</span>
          </button>
          <div className="text-[10px] text-slate-500 text-center mt-1.5 font-medium">
            GeoTIFF, TIFF, PNG, JPEG (Max 100MB)
          </div>
        </div>

        {/* Benchmark Satellite Presets */}
        <div>
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 px-2 flex items-center justify-between">
            <span>Benchmark Scenes</span>
            <LayersIcon className="w-3.5 h-3.5 text-slate-500" />
          </div>
          <div className="space-y-1.5">
            {samples.map((s) => {
              const isSelected = selectedSampleId === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => onSelectSample(s)}
                  className={`w-full text-left p-2.5 rounded-xl border transition group ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/50 text-white'
                      : 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold truncate group-hover:text-white">
                      {s.title}
                    </div>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                        s.modality === 'MULTISPECTRAL'
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : s.modality === 'SAR'
                          ? 'bg-purple-950 text-purple-400 border border-purple-800'
                          : 'bg-blue-950 text-blue-400 border border-blue-800'
                      }`}
                    >
                      {s.modality === 'MULTISPECTRAL' ? '4-Band' : s.modality}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1 line-clamp-1">
                    {s.description}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/50 text-[11px] text-slate-400">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-300">Apple M2 Hybrid</span>
          <span className="text-emerald-400 font-mono">16GB RAM</span>
        </div>
        <div className="text-slate-400 text-[10px] mt-0.5">
          Local NumPy/SciPy • Remote VLM Bridge
        </div>
      </div>
    </aside>
  );
};
