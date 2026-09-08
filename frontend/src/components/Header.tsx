import React from 'react';
import { SatelliteIcon, SparklesIcon } from './Icons';

interface HeaderProps {
  onSelectDemo: (demoId: number) => void;
  activeMode: string;
}

export const Header: React.FC<HeaderProps> = ({ onSelectDemo }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-6 py-3 flex items-center justify-between z-30 sticky top-0">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/30">
          <SatelliteIcon className="w-6 h-6 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold tracking-tight text-white m-0 leading-none">SatQuery AI</h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60">
              SIH26167
            </span>
          </div>
          <p className="text-xs text-slate-400 font-medium tracking-wide">
            ISRO Space Technology • Multimodal Remote Sensing Vision-Language Analyst
          </p>
        </div>
      </div>

      {/* Demo Selector Quick-Bar */}
      <div className="hidden lg:flex items-center space-x-1.5 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
        <span className="text-[11px] font-semibold text-slate-400 px-2 uppercase tracking-wider flex items-center gap-1">
          <SparklesIcon className="w-3.5 h-3.5 text-amber-400" /> Demos:
        </span>
        <button
          onClick={() => onSelectDemo(1)}
          className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          title="Demo 1: Single Image Remote Sensing Captioning & Land-cover"
        >
          1. Captioning
        </button>
        <button
          onClick={() => onSelectDemo(2)}
          className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          title="Demo 2: Water Body Spatial Grounding & Delineation"
        >
          2. Water Grounding
        </button>
        <button
          onClick={() => onSelectDemo(3)}
          className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          title="Demo 3: Multispectral 4-Band NDVI Vegetation Analysis"
        >
          3. NDVI Spectral
        </button>
        <button
          onClick={() => onSelectDemo(4)}
          className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          title="Demo 4: Multitemporal Change Detection (2021 vs 2025)"
        >
          4. Change Detection
        </button>
        <button
          onClick={() => onSelectDemo(5)}
          className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          title="Demo 5: Sentinel-1 SAR & Optical Cross-Modal Reasoning"
        >
          5. Optical + SAR
        </button>
        <button
          onClick={() => onSelectDemo(6)}
          className="px-2.5 py-1 text-xs font-bold rounded-lg bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 transition shadow-sm"
          title="Demo 6: Killer Multi-Step Agentic Analysis Pipeline"
        >
          ★ 6. Killer Demo
        </button>
      </div>

      {/* System Status Pill */}
      <div className="flex items-center space-x-2">
        <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/50 text-emerald-400 text-xs font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Local M2 Hybrid Engine</span>
        </div>
      </div>
    </header>
  );
};
