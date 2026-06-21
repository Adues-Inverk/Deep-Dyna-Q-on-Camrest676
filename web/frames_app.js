const MODE = { COMPARISON: "comparison", K0: "k0", K5: "k5" };

const PATHS = {
  k0Performance:      "/deep_dialog/checkpoints/frames_best/k0/agt_9_performance_records.json",
  k5Performance:      "/deep_dialog/checkpoints/frames_best/k5/agt_9_performance_records.json",
  comparisonDialogue: "/img/frames_k0_vs_k5_dialogue.json",
  k0Cases:            "/img/frames_k0_cases.json",
};

const MEAN_START_EPOCH = 150;

const COLORS = {
  k0:   "#38bdf8",
  k5:   "#a78bfa",
  grid: "rgba(255,255,255,0.08)",
  text: "#d7e2f1",
  muted:"#99a8bd",
};

const state = {
  mode: MODE.COMPARISON,
  k0: null, k5: null,
  comparisonDialogue: null,
  k0Cases: null,
};

const els = {};
function $(id) { return document.getElementById(id); }

function pct(v) {
  if (v === null || v === undefined || isNaN(Number(v))) return "N/A";
  const n = Number(v);
  return Math.abs(n) <= 1 ? `${(n * 100).toFixed(1)}%` : `${n.toFixed(1)}%`;
}
function num(v) {
  if (v === null || v === undefined || isNaN(Number(v))) return "N/A";
  return Number(v).toFixed(2).replace(/\.?0+$/, "");
}
function esc(t) {
  return String(t)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

async function loadJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Cannot load ${path} (${r.status})`);
  return r.json();
}

function series(data, metric) {
  const m = data?.[metric];
  if (!m) return { epochs: [], values: [] };
  const epochs = Object.keys(m).map(Number).sort((a,b) => a-b);
  return { epochs, values: epochs.map(e => Number(m[String(e)])) };
}

function meanFrom(s, start) {
  const vals = s.epochs.map((e,i) => e >= start ? s.values[i] : null).filter(v => v !== null && !isNaN(v));
  return vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null;
}

function best(s, low = false) {
  if (!s.epochs.length) return { epoch: null, value: null };
  return s.epochs.reduce((b,e,i) => {
    const v = s.values[i];
    if (b.value === null) return { epoch: e, value: v };
    return low ? (v < b.value ? {epoch:e,value:v} : b) : (v > b.value ? {epoch:e,value:v} : b);
  }, { epoch: null, value: null });
}

function buildDataset(label, data) {
  const sr    = series(data, "success_rate");
  const turns = series(data, "ave_turns");
  const rew   = series(data, "ave_reward");
  return {
    label, data, sr, turns, rew,
    bestSR:           best(sr),
    meanSRFrom150:    meanFrom(sr, MEAN_START_EPOCH),
    meanTurnsFrom150: meanFrom(turns, MEAN_START_EPOCH),
    meanRewFrom150:   meanFrom(rew, MEAN_START_EPOCH),
  };
}

function allEpochs(...ss) {
  const set = new Set();
  ss.forEach(s => s.epochs.forEach(e => set.add(e)));
  return [...set].sort((a,b) => a-b);
}

function scale(v, i0, i1, o0, o1) {
  if (i1===i0) return (o0+o1)/2;
  return o0 + ((v-i0)/(i1-i0))*(o1-o0);
}

function fmtAxis(metric, v) {
  return metric === "success_rate" ? pct(v) : num(v);
}

function createSvg(title, subtitle, seriesList, metric, height=360) {
  const W=1160, pad={left:78,right:22,top:24,bottom:54};
  const epochs = allEpochs(...seriesList.map(s=>s.series));
  const allV   = seriesList.flatMap(s=>s.series.values).filter(v=>!isNaN(v));
  const minE=epochs[0]??0, maxE=epochs[epochs.length-1]??1;
  let minV, maxV;
  if (metric==="success_rate"){minV=0;maxV=1;}
  else{minV=Math.min(...allV);maxV=Math.max(...allV);if(minV===maxV){minV-=1;maxV+=1;}}
  const cW=W-pad.left-pad.right, cH=height-pad.top-pad.bottom;
  const xFor=e=>scale(e,minE,maxE,pad.left,pad.left+cW);
  const yFor=v=>scale(v,minV,maxV,pad.top+cH,pad.top);

  const gridLines=[];
  for(let i=0;i<=4;i++){
    const y=pad.top+(cH*i)/4, v=maxV-((maxV-minV)*i)/4;
    gridLines.push(
      `<line x1="${pad.left}" y1="${y}" x2="${W-pad.right}" y2="${y}" stroke="${COLORS.grid}" />`,
      `<text x="${pad.left-10}" y="${y+5}" text-anchor="end" fill="${COLORS.muted}" font-size="16">${fmtAxis(metric,v)}</text>`
    );
  }
  const ticks=[];
  const tc=Math.min(8,Math.max(4,Math.round((maxE-minE)/50)+4));
  for(let i=0;i<tc;i++){
    const e=Math.round(scale(i,0,tc-1,minE,maxE)), x=xFor(e);
    ticks.push(
      `<line x1="${x}" y1="${pad.top}" x2="${x}" y2="${pad.top+cH}" stroke="${COLORS.grid}" />`,
      `<text x="${x}" y="${height-16}" text-anchor="middle" fill="${COLORS.muted}" font-size="16">${e}</text>`
    );
  }
  const legend=seriesList.map((item,idx)=>{
    const lx=(item.legendX??pad.left);
    return `<g><rect x="${lx}" y="13" width="14" height="14" rx="4" fill="${item.color}"></rect>
      <text x="${lx+22}" y="25" fill="${COLORS.text}" font-size="16">${esc(item.label)}</text></g>`;
  }).join("");
  const lines=seriesList.map(item=>{
    const path=item.series.epochs.map((e,i)=>
      `${i===0?"M":"L"} ${xFor(e)} ${yFor(item.series.values[i])}`).join(" ");
    return `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join("");

  return `<article class="plot-card">
    <div class="plot-head"><div>
      <h3 class="plot-title">${esc(title)}</h3>
      <p class="plot-subtitle">${esc(subtitle)}</p>
    </div></div>
    <svg class="plot-svg" viewBox="0 0 ${W} ${height}" preserveAspectRatio="none">
      ${gridLines.join("")}${ticks.join("")}${legend}${lines}
    </svg></article>`;
}

// ── Slot rendering for Frames hotel-booking ─────────────────────────────────

function humanKey(k) { return String(k).replace(/_/g," ").trim(); }

function informPhrase(k, v) {
  const key=String(k).toLowerCase(), val=String(v||"").trim();
  if (val.toLowerCase()==="unk"||val==="") return humanKey(k);
  if (key==="taskcomplete") return val.toLowerCase().includes("match") ? "hotel confirmed" : val;
  if (key==="dst_city")     return `in ${val}`;
  if (key==="budget_range") return `${val} budget`;
  if (key==="n_adults")     return `${val} adult${val==="1"?"":"s"}`;
  if (key==="category")     return val;
  if (key==="name")         return val;
  if (key==="price")        return `$${val}/night`;
  if (key==="gst_rating")   return `rated ${val}/10`;
  if (key==="amenities")    return val.replace(/,/g,", ").toLowerCase();
  return `${humanKey(k)}: ${val}`;
}

function requestPhrase(k) {
  const key=String(k).toLowerCase();
  if (key==="name")         return "the hotel name";
  if (key==="price")        return "the nightly price";
  if (key==="gst_rating")   return "the guest rating";
  if (key==="amenities")    return "the amenities";
  if (key==="dst_city")     return "the destination city";
  if (key==="budget_range") return "the budget range";
  if (key==="n_adults")     return "the number of guests";
  if (key==="category")     return "the hotel category";
  return `the ${humanKey(k)}`;
}

function slotList(slots, kind) {
  if (!slots||!Object.keys(slots).length) return "";
  return Object.entries(slots)
    .filter(([k])=>k!=="taskcomplete")
    .map(([k,v])=>kind==="request"?requestPhrase(k):informPhrase(k,v))
    .filter(Boolean).join(", ");
}

function toSentence(turn, isUser) {
  const act    = turn.diaact||"says";
  const inform = slotList(turn.inform_slots,"inform");
  const req    = slotList(turn.request_slots,"request");
  const tc     = turn.inform_slots?.taskcomplete;

  if (isUser) {
    if (act==="thanks") return "Thanks, that's perfect!";
    if (act==="deny")   return "That doesn't work for me.";
    if (act==="inform"&&inform) return `I'm looking for a hotel ${inform}.`;
    if (act==="request"&&req)   return `Could you tell me ${req}?`;
    if (inform) return `I need a hotel ${inform}.`;
    if (req)    return `Could you tell me ${req}?`;
    return `${act.replace(/_/g," ")}.`;
  }
  if (tc&&String(tc).toLowerCase().includes("match"))
    return inform ? `I found a hotel for you — ${inform}.` : "I found a matching hotel!";
  if (act==="no_result"||act==="sorry") return "Sorry, I couldn't find a matching hotel.";
  if (act==="thanks")   return "You're welcome! Enjoy your stay.";
  if (act==="greeting") return "Hello! How can I help you find a hotel?";
  if (act==="closing")  return "Have a great trip!";
  if (inform) return `${inform.charAt(0).toUpperCase()}${inform.slice(1)}.`;
  if (req)    return `Could you tell me ${req}?`;
  return `${act.replace(/_/g," ")}.`;
}

function renderTranscript(hist, container, label) {
  container.innerHTML="";
  if (!hist||!hist.length){container.innerHTML='<div class="empty-state">No transcript.</div>';return;}
  const tmpl=els.turnTemplate;
  for(const turn of hist){
    const isUser=(turn.speaker||"").toLowerCase()==="user";
    const node=tmpl.content.cloneNode(true);
    node.querySelector(".turn-index").textContent   =`Turn ${turn.turn??"?"}`;
    node.querySelector(".turn-speaker").textContent =turn.speaker||"?";
    const cls =isUser?"bubble user-bubble":"bubble agent-bubble";
    const tag =isUser?"User":label;
    const text=toSentence(turn,isUser);
    node.querySelector(".turn-body").innerHTML=
      `<div class="${cls}"><div class="bubble-tag">${esc(tag)}</div><div class="bubble-text">${esc(text)}</div></div>`;
    container.appendChild(node);
  }
}

// ── Render ──────────────────────────────────────────────────────────────────

function renderPlots() {
  const root=els.plotsRoot;
  const {k0,k5}=state;
  if(!k0&&!k5){root.innerHTML='<div class="empty-state">Load a mode to see plots.</div>';return;}

  if(state.mode===MODE.COMPARISON){
    root.innerHTML=
      createSvg("Success rate","k=0 vs k=5 over training episodes",
        [{label:"k=0 (pure DQN)",color:COLORS.k0,series:k0.sr,   legendX:92},
         {label:"k=5 (Dyna-Q)", color:COLORS.k5,series:k5.sr,   legendX:260}],"success_rate")+
      createSvg("Average turns per episode","k=0 vs k=5",
        [{label:"k=0 (pure DQN)",color:COLORS.k0,series:k0.turns,legendX:92},
         {label:"k=5 (Dyna-Q)", color:COLORS.k5,series:k5.turns,legendX:260}],"ave_turns")+
      createSvg("Average reward per episode","k=0 vs k=5",
        [{label:"k=0 (pure DQN)",color:COLORS.k0,series:k0.rew,  legendX:92},
         {label:"k=5 (Dyna-Q)", color:COLORS.k5,series:k5.rew,  legendX:260}],"ave_reward");
    return;
  }

  const ds=state.mode===MODE.K0?k0:k5;
  const label=state.mode===MODE.K0?"k=0 (pure DQN)":"k=5 (Dyna-Q)";
  const col  =state.mode===MODE.K0?COLORS.k0:COLORS.k5;
  root.innerHTML=
    createSvg(`${label} success rate`,"Hotel-booking learning curve",
      [{label,color:col,series:ds.sr,   legendX:92}],"success_rate")+
    createSvg(`${label} average turns`,"Per-episode turns",
      [{label,color:col,series:ds.turns,legendX:92}],"ave_turns")+
    createSvg(`${label} average reward`,"Per-episode reward",
      [{label,color:col,series:ds.rew,  legendX:92}],"ave_reward");
}

function renderTable() {
  const thead=els.resultsTable.querySelector("thead");
  const tbody=els.resultsTable.querySelector("tbody");
  tbody.innerHTML="";

  if(state.mode===MODE.COMPARISON){
    els.tableTitle.textContent="Comparison table";
    thead.innerHTML=`<tr><th>Metric</th><th>k=0 (pure DQN)</th><th>k=5 (Dyna-Q)</th></tr>`;
    const rows=[
      ["Best success rate",                  pct(state.k0.bestSR.value),     pct(state.k5.bestSR.value)],
      ["Best SR achieved at episode",        state.k0.bestSR.epoch??"N/A",   state.k5.bestSR.epoch??"N/A"],
      [`Mean SR (ep ≥ ${MEAN_START_EPOCH})`, pct(state.k0.meanSRFrom150),    pct(state.k5.meanSRFrom150)],
      [`Mean turns (ep ≥ ${MEAN_START_EPOCH})`, num(state.k0.meanTurnsFrom150),num(state.k5.meanTurnsFrom150)],
      [`Mean reward (ep ≥ ${MEAN_START_EPOCH})`, num(state.k0.meanRewFrom150), num(state.k5.meanRewFrom150)],
    ];
    for(const [label,a,b] of rows){
      const tr=document.createElement("tr");
      tr.innerHTML=`<td>${esc(label)}</td><td>${esc(String(a))}</td><td>${esc(String(b))}</td>`;
      tbody.appendChild(tr);
    }
    return;
  }

  const ds=state.mode===MODE.K0?state.k0:state.k5;
  const label=state.mode===MODE.K0?"k=0 (pure DQN)":"k=5 (Dyna-Q)";
  els.tableTitle.textContent=`${label} results`;
  thead.innerHTML=`<tr><th>Metric</th><th>Value</th></tr>`;
  const rows=[
    ["Best success rate",                  pct(ds.bestSR.value)],
    ["Best SR achieved at episode",        ds.bestSR.epoch??"N/A"],
    [`Mean SR (ep ≥ ${MEAN_START_EPOCH})`, pct(ds.meanSRFrom150)],
    [`Mean turns (ep ≥ ${MEAN_START_EPOCH})`, num(ds.meanTurnsFrom150)],
    [`Mean reward (ep ≥ ${MEAN_START_EPOCH})`, num(ds.meanRewFrom150)],
  ];
  for(const [lbl,v] of rows){
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${esc(lbl)}</td><td>${esc(String(v))}</td>`;
    tbody.appendChild(tr);
  }
}

function renderDialogue() {
  const showComp  =state.mode===MODE.COMPARISON&&state.comparisonDialogue;
  const showCases =state.mode===MODE.K0&&state.k0Cases;
  if(!showComp&&!showCases){els.dialoguePanel.classList.add("hidden");return;}
  els.dialoguePanel.classList.remove("hidden");

  if(showComp){
    const c=state.comparisonDialogue;
    els.dialogueTitle.textContent="Same hotel goal — k=0 vs k=5";
    els.leftDialogueColumn.className ="dialogue-column";
    els.rightDialogueColumn.className="dialogue-column";
    els.leftDialogueTitle.textContent ="k=0 (pure DQN)";
    els.rightDialogueTitle.textContent="k=5 (Dyna-Q)";
    const goal=c.goal||{};
    const inf =Object.entries(goal.inform_slots ||{}).map(([k,v])=>`${k}=${v}`).join(", ")||"—";
    const req =Object.keys( goal.request_slots||{}).join(", ")||"—";
    const metaParts=[
      `Goal: inform [${inf}] · request [${req}]`,
      `seed: ${c.seed??"?"}`,
      `k=0 reward/turns: ${c.k0_reward!=null?(c.k0_reward>0?"+":"")+c.k0_reward:"N/A"} / ${c.k0_turns??"N/A"}`,
      `k=5 reward/turns: ${c.k5_reward!=null?(c.k5_reward>0?"+":"")+c.k5_reward:"N/A"} / ${c.k5_turns??"N/A"}`,
    ];
    els.dialogueMeta.textContent=metaParts.join("  •  ");
    renderTranscript(c.k0_history||c.ddq_history||[],els.k0Transcript,"k=0");
    renderTranscript(c.k5_history||c.d3q_history||[],els.k5Transcript,"k=5");
    return;
  }

  const s=state.k0Cases;
  const fail=s.fail_case||{};
  if(!s.success_case){
    els.dialogueTitle.textContent="k=0 · Failure analysis";
    els.leftDialogueColumn.className ="dialogue-column";
    els.rightDialogueColumn.className="dialogue-column fail";
    els.leftDialogueTitle.textContent ="Why k=0 fails";
    els.rightDialogueTitle.textContent="Fail case";
    els.dialogueMeta.textContent=`k=0 achieves 0% success rate on Frames — no success case found in 50 sampled episodes  •  Fail: reward=${fail.reward??"N/A"}, turns=${fail.turns??"N/A"}`;
    els.k0Transcript.innerHTML=`<div class="empty-state" style="text-align:left;padding:1.5rem;line-height:1.7">
      <strong style="color:var(--accent)">Pure DQN gets stuck on hard tasks.</strong><br><br>
      The Frames domain has 10+ cities, 260 hotels, and 6 inform-slots — too many to explore randomly.
      Without world-model planning, the DQN learns a degenerate policy:
      close the booking dialog after 2 turns to minimise per-turn penalties,
      rather than exploring toward a successful booking.<br><br>
      Dyna-Q (k=5) breaks out of this local minimum by generating synthetic
      rollouts through the world model, giving the agent thousands of
      simulated bookings before it ever sees real failure.
    </div>`;
    renderTranscript(fail.history||[],els.k5Transcript,"k=0");
    return;
  }
  const ok=s.success_case;
  els.dialogueTitle.textContent="k=0 · Success vs Fail";
  els.leftDialogueColumn.className ="dialogue-column success";
  els.rightDialogueColumn.className="dialogue-column fail";
  els.leftDialogueTitle.textContent ="Success case";
  els.rightDialogueTitle.textContent="Fail case";
  els.dialogueMeta.textContent=
    `Success: reward=${ok.reward??"N/A"}, turns=${ok.turns??"N/A"}  •  Fail: reward=${fail.reward??"N/A"}, turns=${fail.turns??"N/A"}`;
  renderTranscript(ok.history  ||[],els.k0Transcript,"k=0");
  renderTranscript(fail.history||[],els.k5Transcript,"k=0");
}

function syncButtons() {
  [[els.btnComparison,MODE.COMPARISON],[els.btnK0,MODE.K0],[els.btnK5,MODE.K5]]
    .forEach(([btn,mode])=>btn.classList.toggle("active",state.mode===mode));
  els.modeLabel.textContent=
    state.mode===MODE.COMPARISON?"Comparison":
    state.mode===MODE.K0?"k=0 (DQN)":"k=5 (Dyna-Q)";
  els.sourceLabel.textContent=
    state.mode===MODE.COMPARISON?"k=0 + k=5 + dialogue":
    state.mode===MODE.K0?"k=0 performance + cases":"k=5 performance";
}

function renderAll(){syncButtons();renderPlots();renderTable();renderDialogue();}

function pickCandidate(data){
  if(!data) return data;
  const pool=Array.isArray(data.candidates)?data.candidates:[];
  if(!pool.length) return data;
  return pool[Math.floor(Math.random()*pool.length)];
}

async function loadComparison(){
  state.mode=MODE.COMPARISON;
  const [k0data,k5data,dlg]=await Promise.all([
    loadJson(PATHS.k0Performance),
    loadJson(PATHS.k5Performance),
    loadJson(PATHS.comparisonDialogue),
  ]);
  state.k0=buildDataset("k=0 (pure DQN)",k0data);
  state.k5=buildDataset("k=5 (Dyna-Q)",  k5data);
  state.comparisonDialogue=pickCandidate(dlg);
  state.k0Cases=null;
  renderAll();
}

async function loadK0(){
  state.mode=MODE.K0;
  const [k0data,cases]=await Promise.all([
    loadJson(PATHS.k0Performance),
    loadJson(PATHS.k0Cases),
  ]);
  state.k0=buildDataset("k=0 (pure DQN)",k0data);
  state.k0Cases=cases;
  state.comparisonDialogue=null;
  renderAll();
}

async function loadK5(){
  state.mode=MODE.K5;
  const k5data=await loadJson(PATHS.k5Performance);
  state.k5=buildDataset("k=5 (Dyna-Q)",k5data);
  state.comparisonDialogue=null;
  state.k0Cases=null;
  renderAll();
}

function bindControls(){
  els.btnComparison.addEventListener("click",async()=>{try{await loadComparison();}catch(e){alert(e.message);}});
  els.btnK0.addEventListener("click",        async()=>{try{await loadK0();}        catch(e){alert(e.message);}});
  els.btnK5.addEventListener("click",        async()=>{try{await loadK5();}        catch(e){alert(e.message);}});
}

async function init(){
  Object.assign(els,{
    modeLabel:           $("modeLabel"),
    sourceLabel:         $("sourceLabel"),
    btnComparison:       $("btnComparison"),
    btnK0:               $("btnK0"),
    btnK5:               $("btnK5"),
    plotsRoot:           $("plotsRoot"),
    tableTitle:          $("tableTitle"),
    resultsTable:        $("resultsTable"),
    dialoguePanel:       $("dialoguePanel"),
    dialogueMeta:        $("dialogueMeta"),
    k0Transcript:        $("k0Transcript"),
    k5Transcript:        $("k5Transcript"),
    dialogueTitle:       $("dialogueTitle"),
    leftDialogueTitle:   $("leftDialogueTitle"),
    rightDialogueTitle:  $("rightDialogueTitle"),
    leftDialogueColumn:  $("leftDialogueColumn"),
    rightDialogueColumn: $("rightDialogueColumn"),
    turnTemplate:        $("turnTemplate"),
  });
  bindControls();
  try {
    await loadComparison();
  } catch(e) {
    state.mode=MODE.COMPARISON;
    els.plotsRoot.innerHTML=`<div class="empty-state">Training in progress — check back shortly.<br><br>${esc(e.message)}<br><br>Run <code>python generate_frames_dialogue.py</code> once training completes.</div>`;
    renderAll();
  }
}

window.addEventListener("resize",()=>{if(state.k0||state.k5)renderPlots();});
init();
