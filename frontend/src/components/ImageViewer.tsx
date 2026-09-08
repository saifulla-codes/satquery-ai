import React, { useState } from 'react';
import type { BoundingBox, VisualEvidence, GeoMetadata } from '../types';
import { ZoomInIcon, ZoomOutIcon, CrosshairIcon, Sliders } from './Icons';
import { resolveAssetUrl } from '../services/api';

interface ImageViewerProps {
  primaryImageUrl: string;
  secondaryImageUrl?: string;
  isSplitMode?: boolean;
  layers: VisualEvidence[];
  boundingBoxes: BoundingBox[];
  metadata?: GeoMetadata;
  hoveredBoxId?: string | null;
  onHoverBox?: (boxId: string | null) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  primaryImageUrl,
  secondaryImageUrl,
  isSplitMode = false,
  layers,
  boundingBoxes,
  metadata,
  hoveredBoxId,
  onHoverBox,
}) => {
  const [zoom, setZoom] = useState(1);
  const [splitPos, setSplitPos] = useState(50); // percentage for wiper
  const [layerVisibility, setLayerVisibility] = useState<Record<string, boolean>>({});
  const [layerOpacity, setLayerOpacity] = useState<Record<string, number>>({});
  const [showBoxes, setShowBoxes] = useState(true);

  // Initialize visibility if needed
  const isVisible = (id: string) => (layerVisibility[id] !== undefined ? layerVisibility[id] : true);
  const getOpacity = (layer: VisualEvidence) =>
    layerOpacity[layer.id] !== undefined ? layerOpacity[layer.id] : layer.default_opacity;

  const toggleLayer = (id: string) => {
    setLayerVisibility((prev) => ({
      ...prev,
      [id]: prev[id] !== undefined ? !prev[id] : false,
    }));
  };

  const handleOpacityChange = (id: string, val: number) => {
    setLayerOpacity((prev) => ({ ...prev, [id]: val }));
  };

  return (
    <div className="relative flex-1 bg-slate-950 flex flex-col h-[calc(100vh-65px)] overflow-hidden space-grid border-r border-slate-800">
      {/* Top Toolbar */}
      <div className="absolute top-4 left-4 z-20 flex items-center space-x-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 shadow-xl">
        <button
          onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}
          className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition"
          title="Zoom In"
        >
          <ZoomInIcon className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
          className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition"
          title="Zoom Out"
        >
          <ZoomOutIcon className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom(1)}
          className="px-2 py-1 text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
          title="Reset Zoom"
        >
          {Math.round(zoom * 100)}%
        </button>
        <div className="w-px h-4 bg-slate-700 mx-1" />
        <button
          onClick={() => setShowBoxes((b) => !b)}
          className={`flex items-center space-x-1 px-2 py-1 text-xs font-medium rounded-lg transition ${
            showBoxes
              ? 'bg-cyan-950 text-cyan-400 border border-cyan-800/80'
              : 'text-slate-400 hover:bg-slate-800'
          }`}
          title="Toggle Bounding Boxes"
        >
          <CrosshairIcon className="w-3.5 h-3.5" />
          <span>Boxes ({boundingBoxes.length})</span>
        </button>
      </div>

      {/* Metadata HUD Chip */}
      {metadata && (
        <div className="absolute top-4 right-4 z-20 flex items-center space-x-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 text-xs font-mono shadow-xl text-slate-300">
          <span className="text-cyan-400 font-bold">{metadata.format}</span>
          <span>•</span>
          <span>{metadata.width}×{metadata.height} px</span>
          {metadata.has_nir && (
            <>
              <span>•</span>
              <span className="text-emerald-400 font-bold">NIR Active</span>
            </>
          )}
          {metadata.crs && (
            <>
              <span>•</span>
              <span className="text-slate-400">{metadata.crs}</span>
            </>
          )}
        </div>
      )}

      {/* Main Viewport Container */}
      <div className="flex-1 flex items-center justify-center p-6 overflow-auto relative">
        <div
          style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
          className="relative transition-transform duration-150 ease-out shadow-2xl rounded-lg overflow-hidden border border-slate-700/80 bg-slate-900 select-none max-w-[85%] max-h-[85%]"
        >
          {/* Primary Base Image */}
          <img
            src={resolveAssetUrl(primaryImageUrl)}
            alt="Satellite Scene"
            className="block max-w-full max-h-[70vh] object-contain pointer-events-none"
          />

          {/* Secondary Split Image (Multitemporal / SAR Wiper) */}
          {isSplitMode && secondaryImageUrl && (
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `polygon(${splitPos}% 0%, 100% 0%, 100% 100%, ${splitPos}% 100%)` }}
            >
              <img
                src={resolveAssetUrl(secondaryImageUrl)}
                alt="Paired Satellite Scene"
                className="w-full h-full object-contain"
              />
            </div>
          )}

          {/* Split Wiper Line & Handle */}
          {isSplitMode && secondaryImageUrl && (
            <div
              className="absolute top-0 bottom-0 z-30 w-1 bg-cyan-400 cursor-ew-resize shadow-lg shadow-cyan-400/50"
              style={{ left: `${splitPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-cyan-500 border-2 border-slate-950 flex items-center justify-center text-[10px] font-bold text-slate-950 shadow-md">
                ↔
              </div>
            </div>
          )}

          {/* Visual Evidence Overlays */}
          {layers.map((layer) => {
            if (!isVisible(layer.id)) return null;
            return (
              <img
                key={layer.id}
                src={resolveAssetUrl(layer.layer_url)}
                alt={layer.title}
                style={{ opacity: getOpacity(layer) }}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-200"
              />
            );
          })}

          {/* Spatial Grounding Polygon Boundaries Overlay */}
          {showBoxes && (
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              className="absolute inset-0 w-full h-full pointer-events-none z-10"
            >
              {boundingBoxes.map((box) => {
                if (!box.polygon || box.polygon.length === 0) return null;
                const isHovered = hoveredBoxId === box.id;
                const points = box.polygon.map((p) => `${p.x * 100},${p.y * 100}`).join(' ');
                return (
                  <polygon
                    key={`poly-${box.id}`}
                    points={points}
                    vectorEffect="non-scaling-stroke"
                    className={`transition-all ${
                      isHovered
                        ? 'fill-amber-400/30 stroke-amber-400 stroke-2'
                        : 'fill-cyan-500/20 stroke-cyan-300 stroke-1.5'
                    }`}
                  />
                );
              })}
            </svg>
          )}

          {/* Spatial Grounding Bounding Boxes Overlay */}
          {showBoxes &&
            boundingBoxes.map((box) => {
              const isHovered = hoveredBoxId === box.id;
              return (
                <div
                  key={box.id}
                  onMouseEnter={() => onHoverBox?.(box.id)}
                  onMouseLeave={() => onHoverBox?.(null)}
                  style={{
                    top: `${box.ymin * 100}%`,
                    left: `${box.xmin * 100}%`,
                    width: `${(box.xmax - box.xmin) * 100}%`,
                    height: `${(box.ymax - box.ymin) * 100}%`,
                  }}
                  className={`absolute border-2 transition-all cursor-pointer pointer-events-auto rounded-sm group ${
                    isHovered
                      ? 'border-amber-400 bg-amber-400/20 shadow-lg shadow-amber-400/30 z-20'
                      : 'border-cyan-400/90 bg-cyan-500/10 hover:border-amber-400 hover:bg-amber-400/15'
                  }`}
                >
                  <div className="absolute -top-6 left-0 bg-slate-900/95 backdrop-blur-md px-1.5 py-0.5 rounded text-[10px] font-bold text-cyan-300 border border-slate-700 whitespace-nowrap shadow-md flex items-center space-x-1">
                    <span>{box.label}</span>
                    {box.confidence && (
                      <span className="text-emerald-400 font-mono">
                        {Math.round(box.confidence * 100)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Split Wiper Range Slider (when split mode is active) */}
      {isSplitMode && secondaryImageUrl && (
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-20 w-80 bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-800 shadow-xl flex items-center space-x-3">
          <span className="text-[11px] font-bold text-slate-400 uppercase">T1</span>
          <input
            type="range"
            min={0}
            max={100}
            value={splitPos}
            onChange={(e) => setSplitPos(Number(e.target.value))}
            className="flex-1 accent-cyan-400 cursor-ew-resize h-1.5 bg-slate-800 rounded-lg appearance-none"
          />
          <span className="text-[11px] font-bold text-slate-400 uppercase">T2</span>
        </div>
      )}

      {/* Bottom Layer Tray & Opacity Sliders */}
      {layers.length > 0 && (
        <div className="border-t border-slate-800 bg-slate-900/80 backdrop-blur-md p-3 flex items-center space-x-4 overflow-x-auto z-10 shrink-0">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider shrink-0 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" /> Evidence Layers:
          </span>
          <div className="flex items-center space-x-3">
            {layers.map((l) => (
              <div
                key={l.id}
                className="flex items-center space-x-2 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 shrink-0"
              >
                <button
                  onClick={() => toggleLayer(l.id)}
                  className={`text-xs font-semibold flex items-center space-x-1.5 ${
                    isVisible(l.id) ? 'text-white' : 'text-slate-500 line-through'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      l.layer_type === 'ndvi'
                        ? 'bg-emerald-400'
                        : l.layer_type === 'ndwi'
                        ? 'bg-blue-400'
                        : l.layer_type === 'change_map'
                        ? 'bg-amber-400'
                        : 'bg-purple-400'
                    }`}
                  />
                  <span>{l.title}</span>
                </button>
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={getOpacity(l)}
                  onChange={(e) => handleOpacityChange(l.id, parseFloat(e.target.value))}
                  className="w-16 accent-cyan-400 h-1 bg-slate-800 rounded appearance-none"
                  title="Opacity"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
