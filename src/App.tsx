import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Upload, Volume2, FastForward, Waves, Speaker, Download, ListMusic, FolderPlus, FileAudio, Loader2, Music, Repeat, Settings2, X, SlidersHorizontal, Headphones, Zap } from 'lucide-react';

// Utility to convert AudioBuffer to WAV Blob
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const out = new ArrayBuffer(length);
  const view = new DataView(out);
  const channels = [];
  let sample = 0;
  let offset = 0;
  let pos = 0;

  const setUint16 = (data: number) => { view.setUint16(pos, data, true); pos += 2; };
  const setUint32 = (data: number) => { view.setUint32(pos, data, true); pos += 4; };

  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8); // file length - 8
  setUint32(0x45564157); // "WAVE"
  setUint32(0x20746d66); // "fmt " chunk
  setUint32(16); // length = 16
  setUint16(1); // PCM (uncompressed)
  setUint16(numOfChan);
  setUint32(buffer.sampleRate);
  setUint32(buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
  setUint16(numOfChan * 2); // block-align
  setUint16(16); // 16-bit
  setUint32(0x61746164); // "data" - chunk
  setUint32(length - pos - 4); // chunk length

  for (let i = 0; i < buffer.numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  while (pos < length) {
    for (let i = 0; i < numOfChan; i++) {
      sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // scale to 16-bit signed int
      view.setInt16(pos, sample, true); // write 16-bit sample
      pos += 2;
    }
    offset++; // next source sample
  }

  return new Blob([out], { type: "audio/wav" });
}

class AudioEngine {
  context: AudioContext;
  buffer: AudioBuffer | null = null;
  source: AudioBufferSourceNode | null = null;
  
  dryGain: GainNode;
  wetGain: GainNode;
  convolver: ConvolverNode;
  bassFilter: BiquadFilterNode;
  lowpassFilter: BiquadFilterNode;
  highpassFilter: BiquadFilterNode;
  distortionNode: WaveShaperNode;
  pannerNode: StereoPannerNode;
  masterGain: GainNode;

  isPlaying = false;
  isLooping = false;
  offset = 0;
  lastStartTime = 0;
  currentSpeed = 1;
  loadId = 0;

  onEndedCallback: () => void;

  constructor(onEnded: () => void) {
    this.context = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.onEndedCallback = onEnded;

    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();
    this.convolver = this.context.createConvolver();
    this.bassFilter = this.context.createBiquadFilter();
    this.lowpassFilter = this.context.createBiquadFilter();
    this.highpassFilter = this.context.createBiquadFilter();
    this.distortionNode = this.context.createWaveShaper();
    this.pannerNode = this.context.createStereoPanner();
    this.masterGain = this.context.createGain();

    this.bassFilter.type = 'lowshelf';
    this.bassFilter.frequency.value = 80;

    this.lowpassFilter.type = 'lowpass';
    this.lowpassFilter.frequency.value = 20000;

    this.highpassFilter.type = 'highpass';
    this.highpassFilter.frequency.value = 0;
    
    this.generateImpulseResponse();
    this.initRouting();
  }

  makeDistortionCurve(amount: number) {
    const k = typeof amount === 'number' ? amount : 50,
      n_samples = 44100,
      curve = new Float32Array(n_samples),
      deg = Math.PI / 180;
    for (let i = 0; i < n_samples; ++i) {
      const x = i * 2 / n_samples - 1;
      curve[i] = (3 + k) * x * 20 * deg / (Math.PI + k * Math.abs(x));
    }
    return curve;
  }

  initRouting() {
    this.dryGain.connect(this.bassFilter);
    this.convolver.connect(this.wetGain);
    this.wetGain.connect(this.bassFilter);
    
    this.bassFilter.connect(this.lowpassFilter);
    this.lowpassFilter.connect(this.highpassFilter);
    this.highpassFilter.connect(this.distortionNode);
    this.distortionNode.connect(this.pannerNode);
    this.pannerNode.connect(this.masterGain);
    this.masterGain.connect(this.context.destination);
  }

  generateImpulseResponse() {
    const duration = 3.0;
    const decay = 3.0;
    const sampleRate = this.context.sampleRate;
    const length = sampleRate * duration;
    const impulse = this.context.createBuffer(2, length, sampleRate);
    const left = impulse.getChannelData(0);
    const right = impulse.getChannelData(1);
    
    let lastOutL = 0;
    let lastOutR = 0;
    
    for (let i = 0; i < length; i++) {
      const n = 1 - i / length;
      const noiseL = (Math.random() * 2 - 1) * Math.pow(n, decay);
      const noiseR = (Math.random() * 2 - 1) * Math.pow(n, decay);
      
      lastOutL = lastOutL + 0.15 * (noiseL - lastOutL);
      lastOutR = lastOutR + 0.15 * (noiseR - lastOutR);
      
      left[i] = lastOutL;
      right[i] = lastOutR;
    }
    this.convolver.buffer = impulse;
  }

  async load(file: File) {
    const currentId = ++this.loadId;
    this.stop();
    this.offset = 0;
    const arrayBuffer = await file.arrayBuffer();
    const decoded = await this.context.decodeAudioData(arrayBuffer);
    if (this.loadId !== currentId) throw new Error("Load aborted");
    this.stop();
    this.buffer = decoded;
  }

  setLoop(val: boolean) {
    this.isLooping = val;
    if (this.source) {
      this.source.loop = val;
    }
  }

  play() {
    if (!this.buffer) return;
    if (this.context.state === 'suspended') this.context.resume();
    
    this.source = this.context.createBufferSource();
    this.source.buffer = this.buffer;
    this.source.playbackRate.value = this.currentSpeed;
    this.source.loop = this.isLooping;
    
    this.source.connect(this.dryGain);
    this.source.connect(this.convolver);
    
    this.source.onended = () => {
      if (this.isPlaying && !this.isLooping) {
        const elapsed = (this.context.currentTime - this.lastStartTime) * this.currentSpeed;
        if (this.offset + elapsed >= this.buffer!.duration - 0.1) {
          this.isPlaying = false;
          this.offset = 0;
          this.onEndedCallback();
        }
      }
    };

    this.source.start(0, this.offset);
    this.lastStartTime = this.context.currentTime;
    this.isPlaying = true;
  }

  pause() {
    if (!this.isPlaying || !this.source) return;
    this.isPlaying = false;
    this.source.stop();
    const elapsed = (this.context.currentTime - this.lastStartTime) * this.currentSpeed;
    if (this.isLooping && this.buffer) {
      this.offset = (this.offset + elapsed) % this.buffer.duration;
    } else {
      this.offset += elapsed;
    }
  }

  stop() {
    this.isPlaying = false;
    if (this.source) {
      this.source.onended = null;
      try { this.source.stop(); } catch(e) {}
      this.source.disconnect();
      this.source = null;
    }
  }

  seek(time: number) {
    if (!this.buffer) return;
    const wasPlaying = this.isPlaying;
    if (this.isPlaying) {
      this.pause();
    }
    this.offset = Math.max(0, Math.min(time, this.buffer.duration));
    if (wasPlaying) {
      this.play();
    }
  }

  getCurrentTime() {
    if (!this.buffer) return 0;
    if (!this.isPlaying) return this.offset;
    
    const elapsed = (this.context.currentTime - this.lastStartTime) * this.currentSpeed;
    let rawTime = this.offset + elapsed;
    
    if (this.buffer.duration > 0 && rawTime >= this.buffer.duration) {
      if (this.isLooping) {
        const loops = Math.floor(rawTime / this.buffer.duration);
        this.offset = rawTime - (loops * this.buffer.duration);
        this.lastStartTime = this.context.currentTime;
        rawTime = this.offset;
      } else {
        rawTime = this.buffer.duration;
      }
    }
    
    return Math.min(rawTime, this.buffer.duration);
  }

  getDuration() {
    return this.buffer ? this.buffer.duration : 0;
  }

  setVolume(val: number) {
    const gain = val / 100;
    this.masterGain.gain.setTargetAtTime(gain, this.context.currentTime, 0.05);
  }

  setSpeed(val: number) {
    if (this.isPlaying) {
      const elapsed = (this.context.currentTime - this.lastStartTime) * this.currentSpeed;
      if (this.isLooping && this.buffer) {
        this.offset = (this.offset + elapsed) % this.buffer.duration;
      } else {
        this.offset += elapsed;
      }
      this.lastStartTime = this.context.currentTime;
    }
    this.currentSpeed = val;
    if (this.source) {
      this.source.playbackRate.setTargetAtTime(val, this.context.currentTime, 0.05);
    }
  }

  setReverb(val: number) {
    this.dryGain.gain.setTargetAtTime(1 - val * 0.15, this.context.currentTime, 0.05);
    this.wetGain.gain.setTargetAtTime(val * 0.6, this.context.currentTime, 0.05);
  }

  setBass(val: number) {
    this.bassFilter.gain.setTargetAtTime(val, this.context.currentTime, 0.05);
  }

  setLowpass(val: number) {
    this.lowpassFilter.frequency.setTargetAtTime(val, this.context.currentTime, 0.05);
  }

  setHighpass(val: number) {
    this.highpassFilter.frequency.setTargetAtTime(val, this.context.currentTime, 0.05);
  }

  setDistortion(val: number) {
    if (val === 0) {
      this.distortionNode.curve = null;
    } else {
      this.distortionNode.curve = this.makeDistortionCurve(val * 10);
    }
  }

  setPan(val: number) {
    this.pannerNode.pan.setTargetAtTime(val, this.context.currentTime, 0.05);
  }

  async exportWav(volume: number, speed: number, reverb: number, bass: number, lowpass: number, highpass: number, distortion: number, pan: number): Promise<Blob> {
    if (!this.buffer) throw new Error("No buffer loaded");

    const offlineCtx = new OfflineAudioContext(
      this.buffer.numberOfChannels,
      Math.ceil(this.buffer.length / speed),
      this.buffer.sampleRate
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = this.buffer;
    source.playbackRate.value = speed;

    const dryGain = offlineCtx.createGain();
    const wetGain = offlineCtx.createGain();
    const convolver = offlineCtx.createConvolver();
    const bassFilter = offlineCtx.createBiquadFilter();
    const lowpassFilter = offlineCtx.createBiquadFilter();
    const highpassFilter = offlineCtx.createBiquadFilter();
    const distortionNode = offlineCtx.createWaveShaper();
    const pannerNode = offlineCtx.createStereoPanner();
    const masterGain = offlineCtx.createGain();

    bassFilter.type = 'lowshelf';
    bassFilter.frequency.value = 80;
    bassFilter.gain.value = bass;

    lowpassFilter.type = 'lowpass';
    lowpassFilter.frequency.value = lowpass;

    highpassFilter.type = 'highpass';
    highpassFilter.frequency.value = highpass;

    if (distortion > 0) {
      distortionNode.curve = this.makeDistortionCurve(distortion * 10);
    }

    pannerNode.pan.value = pan;

    dryGain.gain.value = 1 - reverb * 0.15;
    wetGain.gain.value = reverb * 0.6;
    masterGain.gain.value = volume / 100;

    const duration = 3.0;
    const decay = 3.0;
    const length = offlineCtx.sampleRate * duration;
    const impulse = offlineCtx.createBuffer(2, length, offlineCtx.sampleRate);
    const left = impulse.getChannelData(0);
    const right = impulse.getChannelData(1);
    let lastOutL = 0, lastOutR = 0;
    for (let i = 0; i < length; i++) {
      const n = 1 - i / length;
      const noiseL = (Math.random() * 2 - 1) * Math.pow(n, decay);
      const noiseR = (Math.random() * 2 - 1) * Math.pow(n, decay);
      lastOutL = lastOutL + 0.15 * (noiseL - lastOutL);
      lastOutR = lastOutR + 0.15 * (noiseR - lastOutR);
      left[i] = lastOutL;
      right[i] = lastOutR;
    }
    convolver.buffer = impulse;

    source.connect(dryGain);
    source.connect(convolver);
    dryGain.connect(bassFilter);
    convolver.connect(wetGain);
    wetGain.connect(bassFilter);
    bassFilter.connect(lowpassFilter);
    lowpassFilter.connect(highpassFilter);
    highpassFilter.connect(distortionNode);
    distortionNode.connect(pannerNode);
    pannerNode.connect(masterGain);
    masterGain.connect(offlineCtx.destination);

    source.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
    return audioBufferToWav(renderedBuffer);
  }
}

interface SliderProps {
  label: string;
  icon: React.ElementType;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
  format?: (val: number) => string;
}

const Slider = ({ label, icon: Icon, value, min, max, step, onChange, format }: SliderProps) => {
  const percentage = ((value - min) / (max - min)) * 100;
  
  return (
    <div className="flex flex-col gap-3 w-full group">
      <div className="flex justify-between items-center font-medium">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-zinc-800 text-purple-300 rounded-xl group-hover:bg-zinc-900 transition-colors shadow-sm">
            <Icon size={18} />
          </div>
          <span className="tracking-wide text-zinc-800 font-bold">{label}</span>
        </div>
        <span className="text-sm font-mono bg-white/50 text-purple-900 px-3 py-1 rounded-lg border border-white/60 shadow-sm backdrop-blur-md font-bold">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer transition-all"
        style={{
          background: `linear-gradient(to right, #18181b ${percentage}%, rgba(0,0,0,0.1) ${percentage}%)`
        }}
      />
    </div>
  );
};

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};

export default function App() {
  const [engine, setEngine] = useState<AudioEngine | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  const [playlist, setPlaylist] = useState<File[]>([]);
  const [currentTrackIndex, setCurrentTrackIndex] = useState<number>(-1);
  const [isPlaylistMode, setIsPlaylistMode] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  
  const [volume, setVolume] = useState(100);
  const [speed, setSpeed] = useState(1.0);
  const [reverb, setReverb] = useState(0.0);
  const [bass, setBass] = useState(0);
  
  const [lowpass, setLowpass] = useState(20000);
  const [highpass, setHighpass] = useState(0);
  const [distortion, setDistortion] = useState(0);
  const [pan, setPan] = useState(0);

  const [showAdvanced, setShowAdvanced] = useState(false);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDraggingProgress, setIsDraggingProgress] = useState(false);

  const onEndedRef = useRef<() => void>();
  const trackLoadIdRef = useRef(0);

  useEffect(() => {
    onEndedRef.current = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (isPlaylistMode && currentTrackIndex < playlist.length - 1 && !isLooping) {
        const nextIdx = currentTrackIndex + 1;
        setCurrentTrackIndex(nextIdx);
        setTimeout(() => loadTrack(nextIdx, playlist), 100);
      }
    };
  }, [isPlaylistMode, currentTrackIndex, playlist, isLooping]);

  useEffect(() => {
    const newEngine = new AudioEngine(() => {
      if (onEndedRef.current) onEndedRef.current();
    });
    setEngine(newEngine);
    
    return () => {
      newEngine.stop();
      newEngine.context.close();
    };
  }, []);

  // Poll for current time
  useEffect(() => {
    if (!engine || isDraggingProgress) return;
    
    let animationId: number;
    const updateProgress = () => {
      setCurrentTime(engine.getCurrentTime());
      animationId = requestAnimationFrame(updateProgress);
    };
    
    animationId = requestAnimationFrame(updateProgress);
    return () => cancelAnimationFrame(animationId);
  }, [engine, isDraggingProgress]);

  const handleSingleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !engine) return;
    
    engine.stop();
    setIsPlaying(false);
    setIsPlaylistMode(false);
    setPlaylist([file]);
    await loadTrack(0, [file], true);
  };

  const handleFolder = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !engine) return;
    
    const audioFiles = Array.from(files as FileList).filter(f => 
      f.type.startsWith('audio/') || f.name.match(/\.(mp3|wav|flac|m4a|ogg)$/i)
    );
    
    if (audioFiles.length > 0) {
      engine.stop();
      setIsPlaying(false);
      setIsPlaylistMode(true);
      setPlaylist(audioFiles);
      setCurrentTrackIndex(-1);
      setCurrentTime(0);
    }
  };

  const loadTrack = async (index: number, list: File[] = playlist, autoplay = true) => {
    const file = list[index];
    if (!file || !engine) return;

    const currentLoadId = ++trackLoadIdRef.current;

    if (engine.context.state === 'suspended') {
      await engine.context.resume();
    }

    setIsLoading(true);
    setCurrentTrackIndex(index);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    
    try {
      await engine.load(file);
      if (trackLoadIdRef.current !== currentLoadId) return;

      setDuration(engine.getDuration());
      
      // Keep current slider settings
      engine.setVolume(volume);
      engine.setSpeed(speed);
      engine.setReverb(reverb);
      engine.setBass(bass);
      engine.setLowpass(lowpass);
      engine.setHighpass(highpass);
      engine.setDistortion(distortion);
      engine.setPan(pan);
      engine.setLoop(isLooping);

      if (autoplay) {
        engine.play();
        setIsPlaying(true);
      }
    } catch (error: any) {
      if (trackLoadIdRef.current !== currentLoadId) return;
      if (error.message === "Load aborted") return;
      console.error("Error loading audio:", error);
      alert("Failed to load audio file.");
    } finally {
      if (trackLoadIdRef.current === currentLoadId) {
        setIsLoading(false);
      }
    }
  };

  const togglePlay = () => {
    if (!engine || currentTrackIndex === -1 || isLoading) return;
    if (isPlaying) {
      engine.pause();
      setIsPlaying(false);
    } else {
      engine.play();
      setIsPlaying(true);
    }
  };

  const toggleLoop = () => {
    const newLoop = !isLooping;
    setIsLooping(newLoop);
    if (engine) {
      engine.setLoop(newLoop);
    }
  };

  const handleProgressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (engine) {
      engine.seek(time);
    }
  };

  const handleExport = async () => {
    if (!engine || !engine.buffer || currentTrackIndex === -1) return;
    
    setIsExporting(true);
    try {
      const wavBlob = await engine.exportWav(volume, speed, reverb, bass, lowpass, highpass, distortion, pan);
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      const originalName = playlist[currentTrackIndex].name.replace(/\.[^/.]+$/, "");
      a.download = `nightcore_${originalName}.wav`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
      alert("Failed to export track.");
    } finally {
      setIsExporting(false);
    }
  };

  useEffect(() => { if (engine) engine.setVolume(volume); }, [volume, engine]);
  useEffect(() => { if (engine) engine.setSpeed(speed); }, [speed, engine]);
  useEffect(() => { if (engine) engine.setReverb(reverb); }, [reverb, engine]);
  useEffect(() => { if (engine) engine.setBass(bass); }, [bass, engine]);
  useEffect(() => { if (engine) engine.setLowpass(lowpass); }, [lowpass, engine]);
  useEffect(() => { if (engine) engine.setHighpass(highpass); }, [highpass, engine]);
  useEffect(() => { if (engine) engine.setDistortion(distortion); }, [distortion, engine]);
  useEffect(() => { if (engine) engine.setPan(pan); }, [pan, engine]);

  const currentFile = currentTrackIndex >= 0 ? playlist[currentTrackIndex] : null;

  return (
    <div className="min-h-screen bg-white text-zinc-900 font-sans selection:bg-purple-200 py-8 md:py-12 px-4 flex flex-col items-center relative overflow-hidden">
      
      {/* Advanced Menu Toggle (Fixed Top Right) */}
      <button 
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="fixed top-4 right-4 md:top-6 md:right-6 z-50 p-2 md:px-4 md:py-2 bg-white/60 hover:bg-white/90 text-zinc-800 rounded-2xl shadow-sm border border-white/60 transition-all flex items-center gap-2 font-bold text-sm backdrop-blur-md"
      >
        {showAdvanced ? <X size={18} /> : <Settings2 size={18} />}
        <span className="hidden sm:inline">{showAdvanced ? 'Close Advanced' : 'Advanced Effects'}</span>
      </button>

      {/* Cool Background Shapes */}
      <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] bg-purple-200/40 rounded-full mix-blend-multiply filter blur-[100px] animate-pulse pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[60vw] h-[60vw] bg-zinc-200/60 rounded-full mix-blend-multiply filter blur-[120px] pointer-events-none"></div>
      <div className="absolute top-[20%] right-[10%] w-[30vw] h-[30vw] bg-purple-300/30 rounded-full mix-blend-multiply filter blur-[80px] pointer-events-none"></div>

      <div className="max-w-xl w-full relative z-10 flex flex-col items-center gap-8">
        
        {/* Copyright Above */}
        <div className="text-xs font-mono text-zinc-500 tracking-widest uppercase mt-2 font-bold">
          2026 - made by Kakao
        </div>

        {/* Title */}
        <header className="text-center flex flex-col items-center gap-3 relative w-full">
          <h1 className="text-5xl md:text-6xl font-black tracking-tight drop-shadow-sm">
            <span className="text-zinc-900">Nightcore</span> <span className="text-purple-600">Mixer</span>
          </h1>
          <p className="text-zinc-600 font-medium text-sm md:text-base max-w-sm text-center">
            Upload a track or folder, tweak the sliders, and create your own nightcore mix instantly.
          </p>
        </header>

        {/* Image Logo (No borders) */}
        <div className="relative w-48 h-48 md:w-56 md:h-56 flex items-center justify-center">
          <img 
            src="https://files.catbox.moe/xia4uf.png" 
            alt="Logo"
            className={`w-full h-full object-contain transition-transform duration-1000 ${isPlaying ? 'scale-105' : 'scale-100'}`}
            style={{ imageRendering: 'pixelated' }}
          />
        </div>

        {/* Controls - Glassmorphism */}
        <div className="w-full bg-white/30 backdrop-blur-2xl p-6 md:p-8 rounded-[2.5rem] shadow-[0_8px_32px_rgba(0,0,0,0.08)] border border-white/60 flex flex-col gap-8 relative overflow-hidden">
          
          {/* Upload Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 relative z-10">
            <label className={`flex-1 flex items-center justify-center gap-2 bg-white/50 hover:bg-white/70 text-zinc-800 font-bold py-3 px-4 rounded-2xl transition-all border border-white/60 shadow-sm backdrop-blur-md ${isLoading ? 'cursor-wait opacity-70' : 'cursor-pointer'}`}>
              <FileAudio size={20} className="text-purple-600 group-hover:text-purple-800 transition-colors" />
              <span>Upload Single Track</span>
              <input type="file" accept=".mp3,.wav,.flac,.m4a" className="hidden" onChange={handleSingleFile} onClick={(e) => (e.target as HTMLInputElement).value = ''} disabled={isLoading || isExporting} />
            </label>

            <label className={`flex-1 flex items-center justify-center gap-2 bg-white/50 hover:bg-white/70 text-zinc-800 font-bold py-3 px-4 rounded-2xl transition-all border border-white/60 shadow-sm backdrop-blur-md ${isLoading ? 'cursor-wait opacity-70' : 'cursor-pointer'}`}>
              <FolderPlus size={20} className="text-purple-600 group-hover:text-purple-800 transition-colors" />
              <span>Upload Folder (Playlist)</span>
              {/* @ts-ignore - webkitdirectory is non-standard but widely supported */}
              <input type="file" webkitdirectory="true" directory="true" multiple className="hidden" onChange={handleFolder} onClick={(e) => (e.target as HTMLInputElement).value = ''} disabled={isLoading || isExporting} />
            </label>
          </div>

          {/* Playlist Section - ONLY visible if folder uploaded */}
          {isPlaylistMode && playlist.length > 0 && (
            <div className="w-full bg-white/40 rounded-2xl border border-white/60 p-4 max-h-64 overflow-y-auto shadow-inner relative z-10 flex flex-col gap-1">
              <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-2 px-2">
                <ListMusic size={14} /> Playlist ({playlist.length})
              </h3>
              {playlist.map((file, idx) => (
                <button
                  key={`${file.name}-${idx}`}
                  onClick={() => loadTrack(idx, playlist, true)}
                  className={`text-left px-3 py-2 rounded-xl text-sm font-medium truncate transition-all flex items-center gap-2 ${
                    idx === currentTrackIndex
                      ? 'bg-purple-600 text-white shadow-md'
                      : 'hover:bg-white/60 text-zinc-700'
                  }`}
                >
                  {idx === currentTrackIndex && isPlaying ? <Waves size={14} className="animate-pulse" /> : <Music size={14} />}
                  <span className="truncate">{file.name}</span>
                </button>
              ))}
            </div>
          )}

          {/* Play & Progress */}
          <div className="flex flex-col gap-4 relative z-10">
            <div className="flex items-center gap-4">
              
              {/* Loop Button */}
              <button 
                onClick={toggleLoop}
                disabled={!currentFile || isLoading}
                className={`flex-shrink-0 w-12 h-12 flex items-center justify-center rounded-2xl font-bold transition-all duration-300 ${
                  !currentFile || isLoading
                    ? 'bg-zinc-200 text-zinc-400 cursor-not-allowed border border-zinc-300'
                    : isLooping
                      ? 'bg-purple-600 text-white shadow-[0_4px_14px_rgba(147,51,234,0.3)] scale-100 hover:scale-105'
                      : 'bg-white/60 text-zinc-600 hover:bg-white/80 shadow-sm border border-white/60'
                }`}
                title="Toggle Loop"
              >
                <Repeat size={20} />
              </button>

              {/* Play Button */}
              <button 
                onClick={togglePlay}
                disabled={!currentFile || isLoading}
                className={`flex-shrink-0 w-16 h-16 flex items-center justify-center rounded-2xl font-bold text-white transition-all duration-300 ${
                  !currentFile || isLoading
                    ? 'bg-zinc-200 text-zinc-400 cursor-not-allowed border border-zinc-300' 
                    : isPlaying 
                      ? 'bg-zinc-800 hover:bg-zinc-900 shadow-[0_4px_14px_rgba(0,0,0,0.2)] scale-100 hover:scale-105' 
                      : 'bg-purple-600 hover:bg-purple-700 shadow-[0_4px_14px_rgba(147,51,234,0.3)] scale-100 hover:scale-105'
                }`}
              >
                {isPlaying ? <Pause size={28} /> : <Play size={28} className="ml-1" />}
              </button>
              
              <div className="flex flex-col gap-2 w-full min-w-0">
                <div className="text-sm font-bold text-zinc-800 truncate">
                  {isLoading ? 'Loading track...' : (currentFile?.name || 'No track selected')}
                </div>
                <div className="flex flex-col gap-1 w-full min-w-0">
                  <input
                    type="range"
                    min={0}
                    max={duration || 100}
                    step={0.1}
                    value={currentTime}
                    onMouseDown={() => setIsDraggingProgress(true)}
                    onMouseUp={() => setIsDraggingProgress(false)}
                    onTouchStart={() => setIsDraggingProgress(true)}
                    onTouchEnd={() => setIsDraggingProgress(false)}
                    onChange={handleProgressChange}
                    className="w-full h-2 rounded-lg appearance-none cursor-pointer transition-all"
                    style={{
                      background: `linear-gradient(to right, #18181b ${(currentTime / (duration || 1)) * 100}%, rgba(0,0,0,0.1) ${(currentTime / (duration || 1)) * 100}%)`
                    }}
                    disabled={!currentFile || isLoading}
                  />
                  <div className="flex justify-between text-xs font-mono text-purple-900 font-bold">
                    <span>{formatTime(currentTime)}</span>
                    <span>{formatTime(duration)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="h-px w-full bg-zinc-300/50 relative z-10"></div>

          {/* Advanced Sliders (Conditional) */}
          {showAdvanced && (
            <div className="flex flex-col gap-6 relative z-10 bg-white/40 p-5 rounded-2xl border border-white/60 shadow-inner animate-in fade-in slide-in-from-top-4 duration-300">
              <h3 className="text-sm font-bold text-zinc-600 uppercase tracking-wider flex items-center gap-2">
                <Settings2 size={16} /> Advanced Effects
              </h3>
              <Slider 
                label="Lowpass Filter" 
                icon={SlidersHorizontal} 
                value={lowpass} 
                min={200} max={20000} step={100} 
                onChange={setLowpass} 
                format={(v) => `${v} Hz`}
              />
              <Slider 
                label="Highpass Filter" 
                icon={SlidersHorizontal} 
                value={highpass} 
                min={0} max={10000} step={100} 
                onChange={setHighpass} 
                format={(v) => `${v} Hz`}
              />
              <Slider 
                label="Distortion" 
                icon={Zap} 
                value={distortion} 
                min={0} max={10} step={0.1} 
                onChange={setDistortion} 
                format={(v) => `Lev ${v.toFixed(1)}`}
              />
              <Slider 
                label="Stereo Pan" 
                icon={Headphones} 
                value={pan} 
                min={-1} max={1} step={0.05} 
                onChange={setPan} 
                format={(v) => v === 0 ? 'Center' : v < 0 ? `L ${Math.abs(v).toFixed(2)}` : `R ${v.toFixed(2)}`}
              />
            </div>
          )}

          {/* Sliders */}
          <div className="flex flex-col gap-6 relative z-10">
            <Slider 
              label="Volume" 
              icon={Volume2} 
              value={volume} 
              min={0} max={100} step={1} 
              onChange={setVolume} 
              format={(v) => `${v}%`}
            />
            <Slider 
              label="Speed & Pitch" 
              icon={FastForward} 
              value={speed} 
              min={0.5} max={2.0} step={0.01} 
              onChange={setSpeed} 
              format={(v) => `${v.toFixed(2)}x`}
            />
            <Slider 
              label="Reverb" 
              icon={Waves} 
              value={reverb} 
              min={0} max={1.0} step={0.01} 
              onChange={setReverb} 
              format={(v) => `${v.toFixed(2)}`}
            />
            <Slider 
              label="Bass Boost" 
              icon={Speaker} 
              value={bass} 
              min={0} max={24} step={1} 
              onChange={setBass} 
              format={(v) => `Lev ${v}`}
            />
          </div>

          {/* Export Button */}
          <div className="pt-2 relative z-10">
            <button
              onClick={handleExport}
              disabled={!currentFile || isLoading || isExporting}
              className={`w-full flex items-center justify-center gap-2 py-4 px-6 rounded-xl font-bold transition-all duration-300 ${
                !currentFile || isLoading
                  ? 'bg-zinc-200/50 text-zinc-400 cursor-not-allowed border border-zinc-200'
                  : isExporting
                    ? 'bg-purple-600 text-white shadow-[0_4px_14px_rgba(147,51,234,0.3)] cursor-wait'
                    : 'bg-zinc-900 hover:bg-black text-purple-300 shadow-md hover:shadow-lg'
              }`}
            >
              {isExporting ? <Loader2 size={20} className="animate-spin" /> : <Download size={20} />}
              <span>{isExporting ? 'Rendering Audio...' : 'Export Track (.wav)'}</span>
            </button>
          </div>

        </div>

        {/* Copyright Below */}
        <div className="text-xs font-mono text-purple-600 tracking-widest uppercase mb-8 font-bold">
          built from scratch by Kakao
        </div>

      </div>
    </div>
  );
}
