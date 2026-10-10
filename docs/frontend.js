/* Explanatory UI. Playback is illustrative and never launches model training. */
(() => {
'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const D = window.PROJECT_DATA;
if (!D) { $('#data-error').hidden = false; return; }
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const state = { page:'resumen', mode:'audio', process:0, evolution:0, arch:0, topology:'sequential', depth:3, source:'extract', guided:true };
const timers = { process:null, evolution:null, arch:null };
const guides = new Map();
let guideSequence = 0;
const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value).toLocaleString('es-ES');
const sourceNames = {
  extract:'Origen · leer y verificar WAV', split:'Datos · separar personas', folds:'Datos · cinco folds internos',
  duration:'Datos · recortar o rellenar', audio:'Entrada · onda PCM', mel:'Entrada · log-mel',
  binding:'Entrada · seleccionar modalidad', genome:'Arquitectura · muestrear genoma', model:'Arquitectura · seleccionar CNN 1D o 2D',
  conv1d:'Arquitectura · construir CNN 1D', conv2d:'Arquitectura · construir CNN 2D',
  blocks1d:'Arquitectura · residual e Inception 1D', blocks2d:'Arquitectura · residual e Inception 2D',
  initialization:'Evolución · población inicial', fitness:'Evolución · entrenar y validar', crossover:'Evolución · cruce por innovación',
  mutation:'Evolución · mutación', selection:'Evolución · selección y reproducción', monitor:'Evaluación · val global', final:'Evaluación · test global',
};
const sourceEvidence = {
  extract:'Los WAV se leen del disco. El nombre y la carpeta identifican la clase, la persona y la vocal; se comprueba PCM mono de 16 bits.',
  split:'La división 70/10/20 se realiza sobre personas y se estratifica por clase. Todas las grabaciones de una persona permanecen juntas.',
  folds:'StratifiedKFold se aplica solo a las 70 personas de train: 56 entrenan y 14 validan en cada fold.',
  duration:'La longitud común se calcula con el percentil 95 de las duraciones del train global y se redondea hacia arriba a décimas de segundo. El relleno es simétrico; el recorte maximiza energía.',
  audio:'Se divide el PCM16 entre 32.768 y se añade el eje de canal. El resultado por grabación es float32 con forma (1, 286650).',
  mel:'STFT con nperseg=1024, hop=441 y sin padding. Las 64 bandas mel cubren 80–8000 Hz. Se aplica log1p(1000 × mel_power).',
  binding:'configure_vowels_input sustituye los selectores genéricos por vowels y decide arrays_audio o arrays. La modalidad por defecto es audio.',
  genome:'La estructura y los hiperparámetros se muestrean dentro de los límites de configuración, con comprobaciones de viabilidad.',
  model:'El constructor compatible despacha audio a EvolvableCNN1D y spectrogram/image a EvolvableCNN2D.',
  conv1d:'La topología secuencial construye convolución, normalización, activación y MaxPool1d por capa. Residual e Inception tienen constructores específicos.',
  conv2d:'En 2D, se añade un canal al log-mel si hace falta; el pooling reduce frecuencia y tiempo. La salida son dos logits.',
  blocks1d:'El bloque residual suma la ruta principal y el shortcut antes de pooling. Inception concatena ramas de kernels distintos a lo largo del eje de canales.',
  blocks2d:'Los mismos principios se aplican a frecuencia y tiempo: convoluciones 2D, suma residual con proyección automática o ramas Inception concatenadas.',
  initialization:'Inicialización por cuartiles, candidatos de plantillas y un candidato de doble límite incremental. No todos los candidatos iniciales son superficiales.',
  fitness:'Cada fold entrena un modelo nuevo con train, selecciona su estado con validation y devuelve métricas de validation. El test no forma parte de la fitness.',
  crossover:'Los genes se alinean por innovation_id. Los genes compartidos se escogen de uno de los padres y después se reconstruye y valida el genoma.',
  mutation:'Los cambios estructurales y de hiperparámetros se registran en el historial. La nueva estructura debe seguir siendo válida.',
  selection:'La reproducción aplica ranking Pareto o fitness, preserva élites y utiliza selección por torneo, cruce y mutación para los descendientes.',
  monitor:'Las 10 personas de val global monitorizan el mejor checkpoint durante la evolución. Esa métrica se registra aparte de la fitness de los folds internos.',
  final:'Después de fijar el genoma, se entrena de nuevo con 70 personas, se selecciona epoch con las 10 de val y se evalúa el único test global de 20 personas.',
};

function stopPlayback(kind) {
  if (timers[kind]) clearInterval(timers[kind]);
  timers[kind] = null;
  const labels = { process:['#process-play','Reproducir recorrido'], evolution:['#evolution-play','Reproducir evolución'], arch:['#arch-play','Construir con animación'] };
  $(labels[kind][0]).textContent = labels[kind][1];
  $(labels[kind][0]).setAttribute('aria-pressed','false');
  guides.forEach(guide => { if (guide.kind === kind) guide.motion?.pause(); });
}
function pauseAll() {
  Object.keys(timers).forEach(stopPlayback);
  // Pausing also stops the moving signal markers on the diagram.
  $$('.net-edge.lit').forEach(edge=>edge.classList.remove('lit'));
  guides.forEach(guide => guide.motion?.pause());
}
function clearGuide(frame) {
  const guide = guides.get(frame);
  if (!guide) return;
  guide.motion?.cancel();
  cancelAnimationFrame(guide.raf);
  guide.observer.disconnect();
  guide.content.removeEventListener('scroll',guide.update);
  frame.querySelector('.guide-overlay')?.remove();
  frame.classList.remove('guided-active');
  guide.zoomElement.classList.remove('guide-zoom-target');
  guides.delete(frame);
}
function clearAllGuides() { [...guides.keys()].forEach(clearGuide); }
function clearKindGuides(kind) {
  [...guides].filter(([frame,guide])=>guide.kind===kind||!frame.isConnected).forEach(([frame])=>clearGuide(frame));
}

// A bounded camera beat with a steady caption and a leader to the real target.
// Follow its transformed bounds, scrolling and resize; never run a perpetual RAF.
function guideScene(content, target, title, detail, kind, options = {}) {
  [...guides.keys()].filter(frame=>!frame.isConnected).forEach(clearGuide);
  if (!content || !target || !state.guided || content.getBoundingClientRect().width === 0) return;
  let frame = content.parentElement;
  if (!frame.classList.contains('guide-frame')) {
    frame = document.createElement('div');frame.className='guide-frame';
    content.before(frame);frame.append(content);
  }
  clearGuide(frame);
  frame.classList.add('guided-active');
  const markerId='guide-arrow-'+(++guideSequence);
  const overlay=document.createElement('div');overlay.className='guide-overlay';
  overlay.innerHTML=`<div class="guide-caption"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(detail)}</span></div><svg class="guide-leader" aria-hidden="true"><defs><marker id="${markerId}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="currentColor" stroke-width="1.6"/></marker></defs><path class="guide-arrow" marker-end="url(#${markerId})"/><rect class="guide-ring" rx="8"/></svg>`;
  frame.append(overlay);
  const zoomElement=options.zoomElement||target;
  zoomElement.classList.add('guide-zoom-target');
  const guide={content,target,zoomElement,kind,motion:null,raf:0,observer:null,update:null};
  guides.set(frame,guide);
  const update=()=>{
    if(!frame.isConnected||!target.isConnected||frame.getBoundingClientRect().width===0)return;
    const clearance=Math.ceil(overlay.querySelector('.guide-caption').getBoundingClientRect().height)+34;
    if(frame.style.getPropertyValue('--guide-clearance')!==clearance+'px')frame.style.setProperty('--guide-clearance',clearance+'px');
    const f=frame.getBoundingClientRect(),t=target.getBoundingClientRect(),c=overlay.querySelector('.guide-caption').getBoundingClientRect();
    const left=Math.max(5,t.left-f.left-5),right=Math.min(f.width-5,t.right-f.left+5);
    const top=Math.max(c.bottom-f.top+17,t.top-f.top-5),bottom=Math.min(f.height-4,t.bottom-f.top+5);
    const visible=right>left&&bottom>top;
    const leader=overlay.querySelector('svg');leader.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);
    leader.style.visibility=visible?'visible':'hidden';
    const ring=leader.querySelector('rect');
    ring.setAttribute('x',left);ring.setAttribute('y',top);ring.setAttribute('width',Math.max(18,right-left));ring.setAttribute('height',Math.max(18,bottom-top));
    const startX=c.left-f.left+Math.min(c.width-25,Math.max(25,(left+right)/2-(c.left-f.left)));
    const startY=c.bottom-f.top+3,endX=Math.max(left+8,Math.min(right-8,(left+right)/2)),endY=top-7;
    leader.querySelector('.guide-arrow').setAttribute('d',`M${startX} ${startY} C${startX} ${startY+22},${endX} ${endY-26},${endX} ${endY}`);
  };
  guide.update=update;guide.observer=new ResizeObserver(update);guide.observer.observe(frame);
  content.addEventListener('scroll',update,{passive:true});
  update();
  if(!reducedMotion.matches&&!document.hidden) {
    const scale=options.scale||(innerWidth<680?1.055:1.09);
    guide.motion=zoomElement.animate([{scale:'1',offset:0,easing:'cubic-bezier(.16,1,.3,1)'},{scale:String(scale),offset:.28},{scale:String(scale),offset:.72,easing:'cubic-bezier(.16,1,.3,1)'},{scale:'1',offset:1}],{duration:1450,easing:'linear'});
    const started=performance.now();
    const track=now=>{update();if(now-started<1550&&guides.get(frame)===guide)guide.raf=requestAnimationFrame(track);};
    guide.raf=requestAnimationFrame(track);
    overlay.querySelector('.guide-caption').animate([{opacity:.65},{opacity:1}],{duration:250});
  }
}
function togglePlayback(kind, last, render, interval) {
  if (timers[kind]) { stopPlayback(kind); if(kind==='arch')$$('.net-edge.lit').forEach(edge=>edge.classList.remove('lit')); return; }
  if (state[kind] >= last) state[kind] = 0;
  const buttons = {process:'#process-play', evolution:'#evolution-play', arch:'#arch-play'};
  $(buttons[kind]).textContent = 'Pausar';
  $(buttons[kind]).setAttribute('aria-pressed','true');
  timers[kind] = setInterval(() => {
    state[kind]++;
    if (state[kind] >= last) stopPlayback(kind);
    render();
  }, interval);
  render();
}
function goPage() {
  const target = location.hash.slice(1) || 'resumen';
  const page = document.getElementById(target);
  if (!page || !page.classList.contains('page')) { location.hash = 'resumen'; return; }
  pauseAll();
  state.page = target;
  clearAllGuides();
  $$('.page').forEach(el => el.hidden = el.id !== target);
  $$('[data-page]').forEach(a => { if (a.dataset.page === target) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
  $('#page-name').textContent = $(`[data-page="${target}"] span`).textContent;
  if (target === 'arquitectura') renderArchitecture();
  if (target === 'proceso') renderProcess();
  if (target === 'evolucion') renderEvolution();
  if (target === 'fuentes') renderSource();
  drawSignals();
  window.scrollTo({top:0, behavior:'instant'});
}

// Draw genuine project data; the display reductions never change training arrays.
function plotSignal(canvas, mode, compact = false) {
  if (!canvas || canvas.getBoundingClientRect().width === 0) return;
  const box = canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const w = box.width, h = box.height;
  ctx.clearRect(0,0,w,h);
  const pad = mode === 'spectrogram' && !compact ? 40 : 8;
  const plotWidth = w - pad - 8;
  if (mode === 'audio') {
    ctx.strokeStyle = '#2c3b57'; ctx.lineWidth = 1;
    [0.25,0.5,0.75].forEach(y => { ctx.beginPath(); ctx.moveTo(0,h*y); ctx.lineTo(w,h*y); ctx.stroke(); });
    const amplitude = h * .43;
    ctx.strokeStyle = '#83bdff';ctx.lineWidth = Math.max(.9, w / 480 * .65);
    D.example.waveform.forEach((bin, i) => {
      const x = pad + i / (D.example.waveform.length-1) * plotWidth;
      ctx.beginPath();ctx.moveTo(x,h/2-bin.high*amplitude);ctx.lineTo(x,h/2-bin.low*amplitude);ctx.stroke();
    });
    if (!compact) { ctx.fillStyle='#aab8d0';ctx.font='10px Consolas';ctx.fillText('+1',0,12);ctx.fillText('−1',0,h-4); }
  } else {
    const matrix = D.example.spectrogram, bands=matrix.length, frames=matrix[0].length;
    const max = Math.max(...matrix.flat());
    matrix.forEach((row, f) => row.forEach((value, t) => {
      const level = Math.min(1, Math.pow(value / (max || 1), .55));
      const palette = [[12,20,38],[46,51,109],[114,71,166],[218,119,138],[255,217,149]];
      const pos = level*(palette.length-1), a=palette[Math.floor(pos)],b=palette[Math.min(4,Math.floor(pos)+1)],mix=pos%1;
      ctx.fillStyle = `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*mix)).join(',')})`;
      ctx.fillRect(pad+t/frames*plotWidth, (bands-1-f)/bands*(h-5), Math.ceil(plotWidth/frames)+.1, Math.ceil(h/bands)+.1);
    }));
    if (!compact) { ctx.fillStyle='#aab8d0';ctx.font='10px Consolas';ctx.fillText('8 kHz',0,12);ctx.fillText('80 Hz',0,h-4); }
  }
}
function drawSignals() {
  plotSignal($('#hero-signal'),state.mode,true);
  plotSignal($('#data-signal'),state.mode);
  plotSignal($('#process-canvas'), state.process === 4 ? 'spectrogram' : 'audio');
}
function renderRepresentation() {
  const audio = state.mode === 'audio';
  $$('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode)));
  $$('.dim-label').forEach(el=>el.textContent=audio?'1D':'2D');
  $('#hero-representation').textContent=audio?'Onda PCM · 44.100 Hz':'Log-mel · 64 bandas';
  $('#signal-title').textContent=audio?'Amplitud en el tiempo':'Energía por frecuencia y tiempo';
  $('#signal-axis').textContent=audio?'Amplitud PCM / 32.768':'64 bandas mel · escala log1p';
  $('#data-signal').setAttribute('aria-label',audio?'Amplitud de una onda real de la vocal A':'Espectrograma log-mel real de la misma vocal A, frecuencia frente a tiempo');
  $('#sample-caption').textContent=audio?'Datos reales de train. Vista min/máx en 480 intervalos; el array conserva las 286.650 muestras.':'Datos reales de la misma grabación. Vista: media de cada 4 frames; el array conserva 64 × 648 valores.';
  $('#representation-details').innerHTML=audio?`
    <h2>La onda original</h2><p>Se conserva la variación de presión sonora muestreada en el tiempo. Dividir PCM16 entre 32.768 escala la amplitud a [−1, 1), sin normalización estadística ajustada con val o test.</p>
    <div class="representation-spec"><span>Array individual</span><code>(1, 286650) · float32</code></div>
    <div class="representation-spec"><span>Batch de train</span><code>(B, 1, 286650)</code></div>
    <div class="representation-spec"><span>Directorios</span><code>npy_audio/ → arrays_audio/</code></div>
    <div class="representation-spec"><span>Modelo</span><code>Conv1D · batch_size = 2</code></div>
    <div class="representation-spec"><span>Antes de FC</span><code>AdaptiveAvgPool1d(64)</code></div>
    <button class="source-link" data-source="audio">Ver transformación PCM</button>`:`
    <h2>El mapa log-mel</h2><p>La STFT convierte la onda en espectros temporales. Un banco de filtros resume la potencia en 64 bandas mel y una transformación logarítmica comprime su escala.</p>
    <div class="representation-spec"><span>Array individual</span><code>(64, 648) · float32</code></div>
    <div class="representation-spec"><span>Entrada a Conv</span><code>(B, 1, 64, 648)</code></div>
    <div class="representation-spec"><span>Directorios</span><code>npy/ → arrays/</code></div>
    <div class="representation-spec"><span>Modelo</span><code>Conv2D · batch_size = 8</code></div>
    <div class="representation-spec"><span>Antes de FC</span><code>AdaptiveAvgPool2d((4, 8))</code></div>
    <button class="source-link" data-source="mel">Ver cálculo log-mel</button>`;
  drawSignals();
}

const processSteps = [
  { title:'Leer y verificar', short:'WAV y metadatos', source:'extract', heading:'Los archivos ya están en el proyecto', html:'<p>Se recorre <code>new_data/Vowels</code>. El patrón del nombre extrae clase, identificador, vocal y repetición. Se verifica que carpeta y nombre coincidan y que todos los WAV compartan formato.</p><p>El manifiesto registra rutas, duración, muestras y etiquetas: <strong>Control = 0; Patológicas = 1</strong>. Se comprueba que cada persona tenga A, E, I, O, U y tres repeticiones, sin duplicados ni grabaciones ausentes.</p>' },
  { title:'Separar personas', short:'Train / val / test', source:'split', heading:'La unidad de separación es la persona', html:'<p>Con semilla 42 y estratificación por clase, las 100 personas se dividen en <strong>70 train, 10 val y 20 test</strong>. Sus 15 grabaciones viajan juntas. Los conjuntos resultantes contienen 1.050, 150 y 300 audios, respectivamente.</p><p>Sobre las 70 personas de train se crean cinco folds internos. Cada fold contiene 56 personas para entrenar y 14 para validar (840 y 210 grabaciones). Los CSV numeran 0–4; los archivos de entrenamiento, 1–5.</p>' },
  { title:'Igualar duración', short:'6,5 s por grabación', source:'duration', heading:'Se conserva una ventana de tamaño común', html:'<p>Se calcula el percentil 95 de las duraciones del <strong>train global</strong> y se redondea hacia arriba a décimas de segundo. En los artefactos actuales da 6,5 s: 286.650 muestras a 44.100 Hz.</p><ol><li>Si sobra audio, se elige la ventana contigua de máxima energía mediante sumas acumuladas de muestras al cuadrado.</li><li>Si falta audio, se añaden ceros a izquierda y derecha de forma casi simétrica.</li><li>Si ya tiene la longitud correcta, se copia.</li></ol><p>No se remuestrea ni se elimina silencio mediante un detector de voz. Se escribe un WAV procesado conservando PCM16.</p>' },
  { title:'Crear onda PCM', short:'Representación 1D', source:'audio', heading:'Una secuencia de amplitudes, un canal', html:'<p>Cada WAV procesado se convierte con <code>samples.astype(float32) / 32768.0</code>. Se añade un eje de canal: <code>(1, 286650)</code>. Se guarda un archivo por grabación en <code>npy_audio/</code>.</p><p>Esta es la entrada <code>audio</code>, predeterminada. El modelo usa convoluciones 1D sobre el tiempo. El espectrograma también se exporta como alternativa, pero la red no recibe ambos a la vez.</p>' },
  { title:'Crear log-mel', short:'Alternativa 2D', source:'mel', heading:'De tiempo a frecuencia y tiempo', html:'<p>El WAV escalado pasa por STFT: ventana de 1.024 muestras (Hann predeterminada de SciPy), salto de 441 muestras, <code>boundary=None</code> y <code>padded=False</code>. Esto produce 648 frames.</p><p>Se calcula <code>|spectrum|²</code>, se multiplica por 64 filtros triangulares mel normalizados por su suma (80–8.000 Hz) y se aplica <code>log1p(1000 × mel_power)</code>. El resultado <code>(64, 648)</code> se guarda en <code>npy/</code> como float32.</p><p>La transformación es fija por grabación; no aprende parámetros de validación o test. El modo <code>spectrogram</code> activa una CNN 2D con un canal.</p>' },
  { title:'Exportar arrays', short:'X, y y manifiestos', source:'folds', heading:'Los tensores siguen el orden de los CSV', html:'<p>Se escriben X como float32 e y como int64, de manera incremental con <code>open_memmap</code>. Los índices coinciden exactamente con los CSV y las etiquetas binarias. Se exportan los conjuntos globales y los cinco pares train/val internos.</p><p>Ejemplo: <code>X_train_vowels_fold_1.npy</code> contiene 840 grabaciones; <code>X_val_vowels_fold_1.npy</code>, 210. Hay un único <code>X_test_vowels.npy</code> con 300 grabaciones. <strong>No se generan archivos test por fold en vocales.</strong></p><p>Los arrays se guardan en <code>arrays_audio/</code> o <code>arrays/</code>. Las dos modalidades mantienen idéntica separación de personas.</p>' },
  { title:'Cargar y entrenar', short:'DataLoader → CNN', source:'binding', heading:'La modalidad decide qué red se construye', html:'<p><code>test.ipynb</code> llama a <code>configure_vowels_input()</code>. Selecciona el directorio, las dimensiones, el batch y el pooling adaptativo. Los arrays se convierten en tensores PyTorch y se agrupan en DataLoaders.</p><p>Audio: batch 2, pooling final a longitud 64 y caché de folds desactivada. Log-mel: batch 8, pooling a 4 × 8 y caché en RAM. El límite configurado para vocales es de 20 millones de parámetros.</p><p>La función <code>EvolvableCNN()</code> construye CNN 1D o 2D según la modalidad. La clasificación termina en <strong>dos logits</strong>; softmax y argmax se aplican al evaluar.</p><p><a href="#arquitectura">Continuar con la construcción de la arquitectura</a></p>' },
];
function basicFlow(labels) { return `<div class="stage-flow">${labels.map((s,i)=>`${i?'<i aria-hidden="true">→</i>':''}<span>${s}</span>`).join('')}</div>`; }
function personDiagram() {
  let circles='';
  for(let i=0;i<100;i++) { const x=22+(i%20)*29, y=32+Math.floor(i/20)*31;const color=i<70?'#ffbb75':i<80?'#83bdff':'#79dfb1';circles+=`<circle cx="${x}" cy="${y}" r="5" fill="${color}"/>`; }
  return `<svg viewBox="0 0 620 215" role="img" aria-label="100 personas: 70 entrenamiento, 10 validación y 20 test">${circles}<text x="20" y="205" fill="#ffbb75" font-size="12">70 train</text><text x="245" y="205" fill="#83bdff" font-size="12">10 val</text><text x="460" y="205" fill="#79dfb1" font-size="12">20 test</text></svg>`;
}
function durationDiagram() {
  let lines='';
  D.example.waveform.forEach((b,i)=>{if(i%2===0)lines+=`<path d="M${50+i} ${103-b.high*150}V${103-b.low*150}"/>`;});
  return `<svg viewBox="0 0 600 215" role="img" aria-label="Ventana de longitud 286650 muestras y relleno o recorte según duración"><rect x="40" y="25" width="500" height="145" rx="8" fill="#15233a" stroke="#496486"/><g stroke="#83bdff" stroke-width="1.3">${lines}</g><path d="M40 180v8h500v-8" fill="none" stroke="#c7a8fa"/><text x="160" y="209" fill="#c7a8fa" font-size="13">286.650 muestras · 6,5 segundos</text></svg>`;
}
function renderProcess() {
  clearKindGuides('process');
  const p=processSteps[state.process];
  $('#process-steps').innerHTML=processSteps.map((s,i)=>`<button data-process="${i}" ${i===state.process?'aria-current="step"':''}><span class="step-number">${String(i+1).padStart(2,'0')}</span><span><strong>${s.title}</strong><small>${s.short}</small></span></button>`).join('');
  $('#process-counter').textContent=`Paso ${state.process+1} de 7`;
  $('#process-prev').disabled=state.process===0;$('#process-next').disabled=state.process===6;
  $('#process-content').innerHTML=`<h2>${p.heading}</h2>${p.html}<button class="source-link" data-source="${p.source}">Ver este paso en el código</button>`;
  let visual;
  if(state.process===0)visual=basicFlow(['WAV','Persona + clase','Manifiesto']);
  else if(state.process===1)visual=personDiagram();
  else if(state.process===2)visual=durationDiagram();
  else if(state.process===3||state.process===4)visual='<canvas id="process-canvas" width="640" height="230" style="width:100%;height:215px" aria-label="Representación real de la grabación"></canvas>';
  else if(state.process===5)visual=basicFlow(['CSV ordenado','X: float32','y: int64','.npy']);
  else visual=basicFlow([state.mode==='audio'?'(B,1,286650)':'(B,1,64,648)',`CNN ${state.mode==='audio'?'1D':'2D'}`,'2 logits']);
  $('#process-visual').innerHTML=visual;
  drawSignals();
  const processCues=[
    ['.stage-flow span:nth-of-type(2)','Leemos la identidad','Persona, vocal y clase salen del nombre del WAV.'],
    ['svg','Una persona, un conjunto','Sus 15 grabaciones permanecen juntas en train, val o test.'],
    ['svg','Esta es la ventana común','6,5 segundos: se recorta o se rellena, sin remuestrear.'],
    ['canvas','Aquí permanece la onda','La amplitud PCM se escala; no se convierte en imagen.'],
    ['canvas','Ahora vemos frecuencia y tiempo','La misma voz se representa con 64 bandas log-mel.'],
    ['.stage-flow span:nth-of-type(2)','X contiene la entrada','Los valores float32 siguen exactamente el orden del CSV.'],
    ['.stage-flow span:nth-of-type(2)','La modalidad elige la CNN',state.mode==='audio'?'La onda entra en convoluciones 1D.':'El log-mel entra en convoluciones 2D.'],
  ];
  const cue=processCues[state.process];
  guideScene($('#process-visual'),$('#process-visual').querySelector(cue[0]),cue[1],cue[2],'process',{scale:state.process>=1&&state.process<=4?1.045:1.09});
}

function exampleGenome() {
  return {id:'ejemplo-didactico',num_conv_layers:state.depth,num_fc_layers:1,filters:[8,16,32,64].slice(0,state.depth),kernel_sizes:[7,5,3,3].slice(0,state.depth),fc_nodes:[64],activations:['relu'],normalization_type:'batch',dropout_rate:0.2,learning_rate:0.0001,optimizer:'adam',conv_topology:state.topology,
    ...(state.topology==='residual'?{residual_enabled:true,residual_block_size:2,residual_projection:'auto'}:{}),
    ...(state.topology==='inception'?{inception_enabled:true,inception_reduction_ratio:0.5,inception_pool_branch:true}:{})};
}
function architectureSteps() {
  const audio=state.mode==='audio', g=exampleGenome(), nodes=[];
  let length=audio?286650:648, frequency=64, channels=1;
  const shape=()=>audio?`B × ${channels} × ${number(length)}`:`B × ${channels} × ${frequency} × ${length}`;
  const push=(title,sub,description,extra={})=>nodes.push({title,sub,description,width:132,...extra});
  push(audio?'Onda PCM':'Log-mel',shape(),audio?'La onda llega como (B, 1, 286650). B es el número de grabaciones del batch, no el número de personas.':'El array (B, 64, 648) recibe un eje de canal y entra a Conv2D como (B, 1, 64, 648).');
  const dimension=audio?'1D':'2D';
  const doPool=(dropout)=>{length=Math.floor(length/2);if(!audio)frequency=Math.floor(frequency/2);push(`MaxPool${dimension}`,shape(),`Pooling con kernel y stride 2: ${audio?'la longitud temporal se divide':'frecuencia y tiempo se dividen'} entre dos, redondeando hacia abajo.${dropout?' A continuación se aplica dropout fijo de 0,1, porque no es el último bloque.':''}`);};
  for(let i=0;i<g.filters.length;i++) {
    if(state.topology==='residual' && i+1<g.filters.length) {
      const inChannels=channels;
      channels=g.filters[i];
      push(`Conv${dimension} ${i+1}`,shape(),`Convolución con ${channels} filtros y kernel ${g.kernel_sizes[i]}${audio?'':` × ${g.kernel_sizes[i]}`}. Padding conserva tamaño. Normalización batch y ReLU; todavía no se hace pooling.`,{resStart:true,inChannels});
      channels=g.filters[i+1];
      push(`Conv${dimension} ${i+2}`,shape(),`Segunda unidad del bloque residual: ${channels} filtros, kernel ${g.kernel_sizes[i+1]}. Mantiene el tamaño espacial para poder sumar la ruta de entrada.`);
      push('Suma residual',shape(),`Se suma F(x) + shortcut(x) y se aplica la activación final. ${inChannels===channels?'Los canales coinciden y la ruta es identidad.':`Los canales cambian de ${inChannels} a ${channels}; shortcut usa una convolución 1 × 1 de proyección.`} El pooling ocurre después de sumar.`,{resEnd:true,projected:inChannels!==channels});
      i++;doPool(i<g.filters.length-1);
    } else if(state.topology==='inception') {
      channels=g.filters[i];
      push(`Inception ${i+1}`,shape(),`El mismo tensor alimenta cuatro ramas: 1 × 1; reducción 1 × 1 seguida de kernel 3; reducción 1 × 1 seguida de kernel ${Math.max(5,g.kernel_sizes[i])}; y MaxPool stride 1 seguido de 1 × 1. Cada unidad incluye normalización y activación. Las ramas mantienen tamaño y concatenan canales hasta sumar ${channels}.`,{inception:true,width:230,kernel:Math.max(5,g.kernel_sizes[i])});
      doPool(i<g.filters.length-1);
    } else {
      channels=g.filters[i];
      push(`Conv${dimension} ${i+1}`,shape(),`Se materializan ${channels} filtros con kernel ${g.kernel_sizes[i]}${audio?'':` × ${g.kernel_sizes[i]}`}, padding a la mitad del kernel, normalización batch y ReLU. La convolución mantiene ${audio?'el tiempo':'frecuencia y tiempo'}.`,{kernel:g.kernel_sizes[i]});
      doPool(i<g.filters.length-1);
    }
  }
  length=audio?64:8;if(!audio)frequency=4;
  push('AvgPool adaptativo',shape(),audio?'AdaptiveAvgPool1d(64) fija la longitud antes de las densas. Evita que la capa FC dependa de las 286.650 muestras originales.':'AdaptiveAvgPool2d((4,8)) fija frecuencia y tiempo antes de las densas. Reduce y organiza la entrada de tamaño variable.');
  const flat=channels*length*(audio?1:frequency);
  push('Flatten',`B × ${number(flat)}`,`Se aplanan los ejes de características: ${channels} × ${audio?'64':'4 × 8'} = ${number(flat)} valores por grabación. El eje B se conserva.`);
  push('Densa · 64', 'B × 64','Linear transforma las características en 64 nodos. Le siguen normalización batch y ReLU. Con una sola densa oculta no se aplica el dropout del genoma entre capas densas.');
  push('Salida · 2 logits','B × 2','La última Linear produce dos logits: Control y Patológicas. CrossEntropyLoss recibe logits sin softmax; al evaluar, argmax predice la clase y softmax obtiene probabilidades.');
  return nodes;
}
function renderArchitecture() {
  clearKindGuides('arch');
  const nodes=architectureSteps();state.arch=Math.min(state.arch,nodes.length-1);
  const audio=state.mode==='audio';
  $('#arch-model-name').textContent=audio?'EvolvableCNN1D · audio':'EvolvableCNN2D · spectrogram';
  $('#arch-progress').textContent=`Paso ${state.arch+1} de ${nodes.length} · ${state.topology==='sequential'?'Secuencial':state.topology==='residual'?'Residual':'Inception'}`;
  $('#arch-range').max=nodes.length-1;$('#arch-range').value=state.arch;
  $('#arch-prev').disabled=state.arch===0;$('#arch-next').disabled=state.arch===nodes.length-1;
  $('#arch-step-title').textContent=nodes[state.arch].title;
  $('#arch-step-description').innerHTML=`<p>${escapeHTML(nodes[state.arch].description)}</p><p>Forma de salida de este paso: <code>${escapeHTML(nodes[state.arch].sub)}</code>.</p>`;
  $('#arch-source').dataset.source=state.topology==='sequential'?(audio?'conv1d':'conv2d'):(audio?'blocks1d':'blocks2d');
  $('#genome-code').textContent=JSON.stringify(exampleGenome(),null,2);
  const gap=27;let position=22;
  nodes.forEach(n=>{n.x=position;position+=n.width+gap;});
  const width=position+5;
  let markup='<defs><marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#83bdff"/></marker></defs>';
  let resStart=null;
  nodes.forEach((n,i)=>{
    if(i)markup+=`<path class="net-edge ${i<=state.arch&&timers.arch?'lit':''}" d="M${nodes[i-1].x+nodes[i-1].width} 160H${n.x}" ${i<=state.arch?'marker-end="url(#arrowhead)"':''}/>`;
    if(n.resStart)resStart=n.x-12;
    if(n.resEnd&&resStart!==null){markup+=`<path class="net-edge skip-edge" d="M${resStart} 160V56H${n.x+n.width/2}V119" opacity="${i<=state.arch?1:.25}"/><text class="network-annotation" x="${resStart+22}" y="46" opacity="${i<=state.arch?1:.25}">${n.projected?'shortcut · proyección 1 × 1':'shortcut · identidad'}</text>`;resStart=null;}
    const active=i===state.arch, revealed=i<=state.arch;
    markup+=`<g class="net-node ${revealed?(active?'revealed':'built'):'pending'}" data-arch-node="${i}" role="button" tabindex="0" aria-label="Paso ${i+1}: ${escapeHTML(n.title)}, ${escapeHTML(n.sub)}" transform="translate(${n.x},0)">`;
    if(n.inception) {
      markup+=`<rect class="node-box ${active?'active':''}" x="0" y="75" width="${n.width}" height="174" rx="10"/><text x="15" y="96">${n.title}</text>`;
      const branchNames=['1×1','1×1 → 3','1×1 → '+n.kernel,'Pool → 1×1'];
      branchNames.forEach((b,j)=>{const y=115+j*27;markup+=`<path class="net-edge" d="M9 162V${y+9}H26M${n.width-29} ${y+9}H${n.width-9}V162"/><rect x="27" y="${y}" width="${n.width-58}" height="20" rx="3" fill="#293650"/><text x="${n.width/2}" y="${y+14}" text-anchor="middle" class="node-subtitle">${b}</text>`;});
      markup+=`<text x="${n.width/2}" y="237" text-anchor="middle" class="node-subtitle">Concat · ${escapeHTML(n.sub)}</text>`;
    } else {
      const color=i===0?'#83bdff':n.resEnd?'#c7a8fa':i===nodes.length-1?'#79dfb1':'#aab8d0';
      markup+=`<rect class="node-box ${active?'active':''}" x="0" y="119" width="${n.width}" height="84" rx="10"/><path d="M15 135H37" stroke="${color}" stroke-width="2"/><text x="15" y="157">${escapeHTML(n.title)}</text><text x="15" y="181" class="node-subtitle">${escapeHTML(n.sub)}</text>`;
    }
    markup+=`<text x="${n.width/2}" y="278" text-anchor="middle" class="node-subtitle">${i+1}</text></g>`;
  });
  $('#architecture-svg').setAttribute('viewBox',`0 0 ${width} 320`);
  $('#architecture-svg').style.width=`${width}px`;
  $('#architecture-svg').innerHTML=markup;
  $('#architecture-svg').setAttribute('aria-label',`${audio?'CNN 1D':'CNN 2D'} ${state.topology}; paso ${state.arch+1}: ${nodes[state.arch].title}, tensor ${nodes[state.arch].sub}`);
  const scroller=$('.network-scroll'),activeNode=nodes[state.arch];
  scroller.scrollTo({left:Math.max(0,activeNode.x-scroller.clientWidth/2+activeNode.width/2),behavior:reducedMotion.matches?'instant':'smooth'});
  const activeGroup=$(`[data-arch-node="${state.arch}"]`);
  const cueTitle=activeNode.inception?'La entrada se reparte en ramas':activeNode.resEnd?'Aquí se suma el atajo':activeNode.title.startsWith('MaxPool')?'El pooling reduce el tamaño':activeNode.title==='Flatten'?'Estos ejes se aplanan':activeNode.title.startsWith('AvgPool')?'Fijamos el tamaño antes de FC':state.arch===nodes.length-1?'Dos valores, dos clases':state.arch===0?'La red comienza aquí':'Se incorpora esta capa';
  guideScene(scroller,activeGroup.querySelector('.node-box'),cueTitle,`${activeNode.title} · salida ${activeNode.sub}`,'arch',{zoomElement:activeGroup});
}

function miniNet(filters, color='#83bdff', extra='') {
  const xs=[20,78,136,194], counts=[2,3,4,2];
  let shapes='',edges='';
  xs.forEach((x,l)=>{for(let j=0;j<counts[l];j++){const y=30+(j-(counts[l]-1)/2)*13;shapes+=`<circle cx="${x}" cy="${y}" r="3.6" fill="${l===3?'#79dfb1':color}"/>`;if(l<3){for(let k=0;k<counts[l+1];k++){const nextY=30+(k-(counts[l+1]-1)/2)*13;edges+=`<path d="M${x+4} ${y}L${xs[l+1]-4} ${nextY}" stroke="#344868" stroke-width=".7"/>`;}}}});
  return `<svg viewBox="0 0 214 65" aria-hidden="true">${edges}${extra}${shapes}</svg>`;
}
function candidateArchitecture(candidate,color) {
  const filters=candidate.filters.split(' → '), count=filters.length;
  let diagram='';
  filters.forEach((filter,i)=>{
    const x=16+i*178/count, y=31;
    if(i)diagram+=`<path d="M${16+(i-1)*178/count+27} 38H${x}" stroke="#647997"/>`;
    diagram+=`<rect x="${x}" y="${y-6}" width="27" height="26" rx="3" stroke="${color}" fill="#1b2b44"/>`;
    if(candidate.topology==='Inception')diagram+=`<path d="M${x+5} 32h17m-17 5h17m-17 5h17" stroke="${color}"/>`;
    else diagram+=`<path d="M${x+7} 29v18m6-18v18m6-18v18" stroke="${color}"/>`;
  });
  if(candidate.topology==='Residual')diagram+=`<path d="M8 38V9H${16+(count-1)*178/count+14}V24" fill="none" stroke="#c7a8fa"/>`;
  return `<svg viewBox="0 0 210 65" aria-hidden="true">${diagram}</svg>`;
}
function offspringArchitecture(mutated) {
  const audio=state.mode==='audio',dim=audio?'1D':'2D';
  const blocks=[['Conv'+dim,'8 · k5'],['MaxPool','stride 2'],['Conv'+dim,'16 · k3'],['MaxPool','stride 2']];
  if(mutated)blocks.push(['Conv'+dim,'32 · k3','new'],['MaxPool','stride 2','new']);
  blocks.push(['AvgPool',audio?'longitud 64':'4 × 8'],['Densa','64 nodos'],['Salida','2 logits']);
  let lines='';
  blocks.forEach((b,i)=>{
    const x=5+i*97;
    if(i)lines+=`<path d="M${x-9} 51H${x}" class="net-edge"/>`;
    lines+=`<g class="${b[2]?'new-convolution':''}" data-offspring-block="${i}"><rect x="${x}" y="18" width="88" height="66" rx="7" fill="${b[2]?'#36264e':'#192a41'}" stroke="${b[2]?'#c7a8fa':'#4a6186'}"/><text x="${x+44}" y="44" text-anchor="middle" fill="#edf3ff" font-size="12">${b[0]}</text><text x="${x+44}" y="64" text-anchor="middle" fill="${b[2]?'#e0c6ff':'#aab8d0'}" font-size="11">${b[1]}</text>${b[2]?`<text x="${x+44}" y="102" text-anchor="middle" fill="#c7a8fa" font-size="11">Añadido</text>`:''}</g>`;
  });
  return `<div class="offspring-diagram"><svg viewBox="0 0 ${blocks.length*97} 114" style="width:${blocks.length*97}px" role="img" aria-label="Arquitectura del descendiente con ${mutated?'tres':'dos'} capas convolucionales, pooling, densa y dos logits">${lines}</svg></div>`;
}
const evolutionSteps=[
  {phase:'Población',title:'Cada candidato es un genoma distinto',source:'initialization',text:'<p>El motor crea 20 candidatos con distintas profundidades y topologías. Se combinan genomas aleatorios, plantillas y un individuo de doble límite incremental. Aquí representamos seis para que puedas seguirlos visualmente.</p><p>Los números de filtros mostrados pertenecen a esta demostración. No se asignan métricas hasta entrenar y validar cada candidato.</p>'},
  {phase:'Evaluación',title:'Entrenar cada arquitectura, cinco veces',source:'fitness',text:'<p>Cada genoma se materializa en un modelo nuevo para cada fold interno. Train actualiza los pesos y validation selecciona el checkpoint. Con vocales, cada evaluación interna usa 840 grabaciones de train y 210 de validation.</p><p>La fitness combina los cinco F1 de validation. También se registra el tiempo y el número de parámetros. El paralelismo depende de <code>fold_parallel_workers</code>; no se asume que cinco tareas siempre corran a la vez.</p>'},
  {phase:'Selección',title:'Conservar calidad sin ignorar el coste',source:'selection',text:'<p>Pareto busca maximizar fitness y minimizar tiempo y parámetros. Un candidato domina a otro si no es peor en ningún objetivo y mejora al menos uno. La distancia de crowding mantiene alternativas en el frente.</p><p>Las élites globales se conservan y los padres se eligen por torneo. En esta demostración A se marca como élite y A/B como padres; son roles ilustrativos, no ganadores medidos.</p>'},
  {phase:'Cruce',title:'Combinar genes que se corresponden',source:'crossover',text:'<p>El cruce alinea genes homólogos por <code>innovation_id</code>. Un gen compartido puede venir de cualquiera de los padres; los genes disjuntos del dominante se conservan y algunos del otro pueden incorporarse.</p><p>En el ejemplo, el hijo toma filtros de A y el kernel de B. Después se reconstruye el genoma y se valida. Se heredan decisiones arquitectónicas, no una mezcla visual de pesos entrenados.</p>'},
  {phase:'Mutación',title:'Introducir una modificación estructural',source:'mutation',text:'<p>El hijo añade una capa convolucional de 32 filtros. Esta es una posible mutación de profundidad, no una mejora garantizada. Otras mutaciones cambian filtros, kernels, topología o hiperparámetros.</p><p>El historial registra el cambio. Los límites de crecimiento, dimensiones tras pooling y presupuesto de parámetros restringen qué modificaciones son válidas.</p>'},
  {phase:'Validación',title:'Comprobar que la red puede construirse',source:'genome',text:'<p>El validador ajusta listas y restricciones del genoma. Comprueba kernels, canales, profundidad y el coste estimado. En 2D, el eje de frecuencia puede agotarse antes que el tiempo.</p><p>La nueva capa cambia el tensor de salida y la entrada a las densas. El pooling adaptativo fija el tamaño antes de Flatten. Un candidato incompatible se corrige o se rechaza.</p>'},
  {phase:'Nueva evaluación',title:'El descendiente debe demostrar su valor',source:'fitness',text:'<p>El nuevo modelo se entrena y valida de nuevo en los cinco folds internos. Solo entonces tiene una fitness comparable. Una arquitectura más grande también puede ser más lenta o peor.</p><p>La élite puede reutilizar su evaluación almacenada. El test global de 20 personas sigue reservado. La animación no muestra un aumento de F1 porque aquí no se ha ejecutado entrenamiento.</p>'},
  {phase:'Siguiente generación',title:'Repetir el ciclo con nuevos candidatos',source:'selection',text:'<p>La población siguiente reúne élites, descendientes y el candidato especial de doble límite. El motor adapta la tasa de mutación y actualiza los límites incrementales para seguir explorando.</p><p>Al terminar la búsqueda, se fija el genoma ganador por fitness, se entrena de nuevo y se realiza la evaluación final. <a href="#protocolo">Ver el protocolo de test reservado</a>.</p>'},
];
function renderEvolution() {
  clearKindGuides('evolution');
  const index=state.evolution,step=evolutionSteps[index];
  $('#evolution-counter').textContent=`Etapa ${index+1} de 8`;
  $('#evolution-prev').disabled=index===0;$('#evolution-next').disabled=index===7;
  $('#evolution-phase').textContent=step.phase;$('#evolution-heading').textContent=step.title;
  $('#evolution-description').innerHTML=step.text;$('#evolution-source').dataset.source=step.source;
  $('#population-title').textContent=index===7?'Población siguiente':index>=3?'Padres y descendiente':'Población inicial';
  const candidates=[{id:'A',topology:'Secuencial',filters:'8 → 16'},{id:'B',topology:'Secuencial',filters:'16 → 32'},{id:'C',topology:'Residual',filters:'8 → 16 → 32'},{id:'D',topology:'Inception',filters:'16 → 32'},{id:'E',topology:'Secuencial',filters:'8 → 16 → 32 → 64'},{id:'F',topology:'Residual',filters:'16 → 32'}];
  if(index===7)candidates.splice(1,1,{id:'Hijo',topology:'Secuencial',filters:'8 → 16 → 32'});
  $('#population-grid').innerHTML=candidates.map((c,i)=>{
    let cls='';if(index>=2&&index<7){if(i===0)cls='elite';else if(i===1)cls='selected';else cls='faded';}if(index===1||index===6)cls='selected';if(index===7&&i===1)cls='mutated';
    const annotation=index===0?'Sin evaluar':index===1?'5 folds · en demostración':index===2&&i===0?'Élite ilustrativa':index===7&&i===1?'Nuevo descendiente':'Ejemplo sin métricas';
    return `<div class="candidate ${cls}"><strong>${c.id} · ${c.topology}</strong>${candidateArchitecture(c,i===1?'#c7a8fa':'#83bdff')}<span class="candidate-meta">Filtros ${c.filters}<br>${annotation}</span></div>`;
  }).join('');
  $('#offspring-view').innerHTML=index>=3?`
    <div class="gene-row"><span>Padre A</span><span class="gene">filtros 8, 16</span><span class="gene">kernel 7, 5</span></div>
    <div class="gene-row"><span>Padre B</span><span class="gene parent-b">filtros 16, 32</span><span class="gene parent-b">kernel 5, 3</span></div>
    <div class="gene-row"><strong>Hijo</strong><span class="gene">8</span><span class="gene">16</span>${index>=4?'<span class="gene changed">+ 32</span>':''}<span class="gene parent-b">kernel 5, 3${index>=4?', 3':''}</span></div>
    ${offspringArchitecture(index>=4)}
    <p class="caption">${index>=4?'Añadir una capa altera la arquitectura; falta medir su efecto.':'Cruce ilustrativo de decisiones del genoma.'}</p>`:'';
  $('#evolution-track').innerHTML=evolutionSteps.map((s,i)=>`<button data-evolution="${i}" ${index===i?'aria-current="step"':''}>${i+1}. ${s.phase}</button>`).join('');
  const cues=[
    ['#population-grid','.candidate','Cada red empieza con un genoma','Cambian filtros, profundidad y topología entre candidatos.'],
    ['#population-grid','.candidate','Este candidato pasa por cinco folds','Train ajusta pesos; validation determina su fitness.'],
    ['#population-grid','.elite','La élite se conserva','Este rol es ilustrativo; aquí no se inventan métricas.'],
    ['#offspring-view','.gene-row:nth-child(3) .parent-b','Este kernel viene del padre B','El hijo combina genes homólogos de los dos padres.'],
    ['.offspring-diagram','[data-offspring-block="4"]','La mutación añade esta capa','Nueva convolución: 32 filtros, kernel 3.'],
    ['.offspring-diagram','[data-offspring-block="6"]','La nueva red debe seguir siendo válida','El pooling adaptativo fija la entrada a las capas densas.'],
    ['#population-grid','.candidate','Se vuelve a entrenar y validar','Una arquitectura mayor no garantiza una fitness mejor.'],
    ['#population-grid','.mutated','El descendiente entra en la población','La siguiente generación conserva élites y explora variantes.'],
  ];
  const cue=cues[index],content=$(cue[0]),target=content?.querySelector(cue[1]);
  guideScene(content,target,cue[2],cue[3],'evolution',{scale:index===4?1.14:1.07});
  if(content?.classList.contains('offspring-diagram')&&target){
    // Reparenting the scroll surface can reset its offset; focus after wrapping.
    content.scrollTo({left:Math.max(0,Number(target.dataset.offspringBlock)*97+49-content.clientWidth/2),behavior:'instant'});
    guides.get(content.parentElement)?.update();
  }
}

function renderProtocol() {
  const generic=$('#protocol-select').value==='generic';
  $('#protocol-content').innerHTML=generic?`
    <h2>Dos fuentes de datos, separadas por fase</h2><div class="protocol-path"><h3>Búsqueda · sintéticos</h3><p><code>files_syn_all_N / 40_1e5_N</code>: cinco pares train/validation para evaluar genomas. La fitness usa validation; no incorpora test sintético.</p></div><div class="protocol-path"><h3>Evaluación final · reales</h3><p><code>files_real_N / real_N</code>: el genoma seleccionado se reentrena desde cero en cinco particiones reales. Validation elige epoch y cada test aporta las métricas finales. Los pesos de la búsqueda no se reutilizan por defecto.</p></div><p class="notice">Este es el protocolo genérico del paquete, no el experimento de vocales de test.ipynb. Sin un manifiesto que pruebe rotación exhaustiva de sujetos de test, estas cinco particiones no deben describirse como una validación cruzada clásica por paciente.</p>`:`
    <h2>El test global permanece reservado</h2><div class="split-bar" role="img" aria-label="70 personas train, 10 val y 20 test"><span class="train-part">70 train</span><span class="val-part">10 val</span><span class="test-part">20 test</span></div>
    <p class="caption">Cada grupo está equilibrado por clase. Todas las vocales y repeticiones de una persona permanecen juntas.</p>
    <div class="table-scroll"><table class="protocol-table"><caption class="sr-only">Uso de las particiones en el experimento de vocales</caption><thead><tr><th>Conjunto</th><th>Personas / audios</th><th>Para qué se usa</th></tr></thead><tbody><tr><td>Train global</td><td>70 / 1.050</td><td>Contiene los cinco folds internos; reentrena el ganador al final.</td></tr><tr><td>Train de cada fold</td><td>56 / 840</td><td>Actualiza los pesos de cada candidato.</td></tr><tr><td>Val de cada fold</td><td>14 / 210</td><td>Selecciona checkpoint; sus F1 se promedian como fitness.</td></tr><tr><td>Val global</td><td>10 / 150</td><td>Monitoriza el mejor checkpoint durante la búsqueda; selecciona epoch en el entrenamiento final.</td></tr><tr><td>Test global</td><td>20 / 300</td><td>Produce métricas tras fijar arquitectura y checkpoint final.</td></tr></tbody></table></div>
    <div class="protocol-path"><h3>Durante la búsqueda</h3><p>Los cinco folds se forman solo con las 70 personas de train. Cada persona de train valida una vez en los folds internos. La val global se registra por separado con <code>monitor_val_f1</code> y no sustituye la fitness media de validación interna.</p><button class="source-link" data-source="monitor">Ver monitorización global</button></div>
    <div class="protocol-path"><h3>Después de elegir el genoma</h3><p>El notebook crea loaders globales de train, val y test. <code>evaluate_single_fold()</code> construye un modelo nuevo, lo entrena con 1.050 grabaciones y selecciona el mejor epoch con las 150 de val. Restaura ese estado y calcula las métricas con las 300 de test, sin reentrenar con test.</p><p>El resultado se guarda como <code>artifacts/test_vowels_audio/vowels_final_test.json</code> o en <code>test_vowels_spectrogram/</code> según la modalidad. El protocolo declarado es <code>single_global_person_held_out_test</code>: un único test global, no cinco tests.</p><button class="source-link" data-source="final">Ver evaluación final</button></div>`;
}
function renderSource() {
  const source=D.sources[state.source];
  $('#source-select').value=state.source;
  $('#source-path').textContent=source.path;
  $('#source-location').textContent=source.cell?`Celda ${source.cell} (contando desde 1)`:`Desde línea ${source.line} · SHA ${source.sha256}`;
  $('#source-code').textContent=source.code;
  $('#source-evidence').textContent=sourceEvidence[state.source];
}
function changeMode(mode) {
  if(state.mode===mode)return;
  pauseAll();state.mode=mode;state.arch=0;
  renderRepresentation();renderArchitecture();renderProcess();renderEvolution();
}
document.addEventListener('click',async event=>{
  const guideToggle=event.target.closest('.guide-toggle');
  if(guideToggle){
    state.guided=!state.guided;clearAllGuides();
    $$('.guide-toggle').forEach(button=>button.setAttribute('aria-pressed',String(state.guided)));
    if(state.page==='arquitectura')renderArchitecture();
    if(state.page==='proceso')renderProcess();
    if(state.page==='evolucion')renderEvolution();
    return;
  }
  const mode=event.target.closest('[data-mode]');if(mode){changeMode(mode.dataset.mode);return;}
  const source=event.target.closest('[data-source]');if(source){state.source=source.dataset.source;renderSource();location.hash='fuentes';return;}
  const step=event.target.closest('[data-process]');if(step){stopPlayback('process');state.process=Number(step.dataset.process);renderProcess();return;}
  const evo=event.target.closest('[data-evolution]');if(evo){stopPlayback('evolution');state.evolution=Number(evo.dataset.evolution);renderEvolution();return;}
  const node=event.target.closest('[data-arch-node]');if(node){stopPlayback('arch');state.arch=Number(node.dataset.archNode);renderArchitecture();}
});
$('#architecture-svg').addEventListener('keydown',event=>{
  const node=event.target.closest('[data-arch-node]');
  if(node&&(event.key==='Enter'||event.key===' ')){event.preventDefault();stopPlayback('arch');state.arch=Number(node.dataset.archNode);renderArchitecture();}
});
$('#process-play').addEventListener('click',()=>togglePlayback('process',6,renderProcess,4000));
$('#process-prev').addEventListener('click',()=>{stopPlayback('process');state.process=Math.max(0,state.process-1);renderProcess();});
$('#process-next').addEventListener('click',()=>{stopPlayback('process');state.process=Math.min(6,state.process+1);renderProcess();});
$('#evolution-play').addEventListener('click',()=>togglePlayback('evolution',7,renderEvolution,3800));
$('#evolution-prev').addEventListener('click',()=>{stopPlayback('evolution');state.evolution=Math.max(0,state.evolution-1);renderEvolution();});
$('#evolution-next').addEventListener('click',()=>{stopPlayback('evolution');state.evolution=Math.min(7,state.evolution+1);renderEvolution();});
$('#evolution-reset').addEventListener('click',()=>{stopPlayback('evolution');state.evolution=0;renderEvolution();});
$('#arch-play').addEventListener('click',()=>togglePlayback('arch',architectureSteps().length-1,renderArchitecture,Number($('#speed').value)));
$('#arch-prev').addEventListener('click',()=>{stopPlayback('arch');state.arch=Math.max(0,state.arch-1);renderArchitecture();});
$('#arch-next').addEventListener('click',()=>{stopPlayback('arch');state.arch=Math.min(architectureSteps().length-1,state.arch+1);renderArchitecture();});
$('#arch-reset').addEventListener('click',()=>{stopPlayback('arch');state.arch=0;renderArchitecture();});
$('#arch-range').addEventListener('input',()=>{stopPlayback('arch');state.arch=Number($('#arch-range').value);renderArchitecture();});
$('#topology').addEventListener('change',()=>{stopPlayback('arch');state.topology=$('#topology').value;state.arch=0;renderArchitecture();});
$('#depth').addEventListener('change',()=>{stopPlayback('arch');state.depth=Number($('#depth').value);state.arch=0;renderArchitecture();});
$('#speed').addEventListener('change',()=>{if(timers.arch){stopPlayback('arch');togglePlayback('arch',architectureSteps().length-1,renderArchitecture,Number($('#speed').value));}});
$('#protocol-select').addEventListener('change',renderProtocol);
$('#source-select').addEventListener('change',()=>{state.source=$('#source-select').value;renderSource();});
$('#copy-genome').addEventListener('click',async()=>{
  const button=$('#copy-genome');
  try{await navigator.clipboard.writeText(JSON.stringify(exampleGenome(),null,2));button.textContent='JSON copiado';}
  catch{const range=document.createRange();range.selectNodeContents($('#genome-code'));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent='Seleccionado · copiar manualmente';}
  setTimeout(()=>button.textContent='Copiar JSON',2600);
});
$$('[data-config]').forEach(el=>el.textContent=number(D.defaults[el.dataset.config]));
$('#template-fraction').textContent=number(D.defaults.architecture_template_seed_fraction*100)+' %';
$('#source-select').innerHTML=Object.keys(D.sources).map(key=>`<option value="${key}">${sourceNames[key]||key}</option>`).join('');
$('.sidebar-bottom p').textContent='Snapshot · '+new Intl.DateTimeFormat('es-ES',{day:'numeric',month:'long',year:'numeric',timeZone:'Europe/Madrid'}).format(new Date(D.exported_at));
$('#hero-network').innerHTML=miniNet([8,16,32]);
renderRepresentation();renderProcess();renderArchitecture();renderEvolution();renderProtocol();renderSource();goPage();
window.addEventListener('hashchange',goPage);
window.addEventListener('resize',drawSignals);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pauseAll();});
reducedMotion.addEventListener('change',()=>{
  clearAllGuides();
  if(state.page==='arquitectura')renderArchitecture();
  if(state.page==='proceso')renderProcess();
  if(state.page==='evolucion')renderEvolution();
});
})();
