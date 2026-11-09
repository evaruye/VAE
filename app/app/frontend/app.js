const $ = (id) => document.getElementById(id);
let rawSchema = {
  ready:false, ui_ready:false, model_ready:false,
  materials:{oxidizer:[], reducer:[], third_monomer:[]}, material_labels:{}, material_counts:{}
};

const families = {
  oxidizer:{container:'oxidizerSlots', slots:2, typePrefix:'oxidizer_type', pctPrefix:'oxidizer_pct', label:'Oxidizer'},
  reducer:{container:'reducerSlots', slots:3, typePrefix:'reducer_type', pctPrefix:'reducer_pct', label:'Reducer'},
  third_monomer:{container:'thirdMonomerSlots', slots:7, typePrefix:'third_monomer_type', pctPrefix:'third_monomer_pct', label:'Functional comonomer'}
};

function escapeHtml(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function setupTabs(){document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-pane').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$('tab-'+btn.dataset.tab).classList.add('active');}));}
function materialLabel(rawName){return rawSchema.material_labels?.[rawName]||rawName;}

function renderFamily(key){
  const f=families[key], list=rawSchema.materials?.[key]||[];
  let html='<div class="slot-header"><span>No.</span><span>Material</span><span>Dosage (wt%)</span><span></span></div>';
  for(let i=1;i<=f.slots;i++){
    const options=list.map(x=>`<option value="${escapeHtml(x)}">${escapeHtml(materialLabel(x))}</option>`).join('');
    html+=`<div class="slot-row"><span class="slot-index">${String(i).padStart(2,'0')}</span><div class="material-input"><select class="material-name" id="${f.typePrefix}${i}"><option value="">Select ${f.label.toLowerCase()}</option>${options}</select></div><input class="material-pct" id="${f.pctPrefix}${i}" type="number" min="0" step="0.001" placeholder="0.000"/><button class="clear-slot" data-type="${f.typePrefix}${i}" data-pct="${f.pctPrefix}${i}" title="Clear this slot">×</button></div>`;
  }
  $(f.container).innerHTML=html;
  $(f.container).querySelectorAll('select,input').forEach(x=>x.addEventListener('input',updateMaterialCount));
  $(f.container).querySelectorAll('.clear-slot').forEach(b=>b.addEventListener('click',()=>{$(b.dataset.type).value='';$(b.dataset.pct).value='';updateMaterialCount();}));
}

function renderPropertySlots(){
  let em='<div class="slot-header property-3"><span>No.</span><span>Dosage (wt%)</span><span>EO number</span><span></span></div>';
  for(let i=1;i<=4;i++) em+=`<div class="slot-row property-3"><span class="slot-index">${String(i).padStart(2,'0')}</span><input id="emulsifier_pct${i}" type="number" min="0" step="0.001" placeholder="0.000"/><input id="emulsifier_eo${i}" type="number" min="0" step="0.1" placeholder="EO"/><button class="clear-slot" data-clear="emulsifier:${i}">×</button></div>`;
  $('emulsifierPropertySlots').innerHTML=em;

  let pc='<div class="slot-header property-4"><span>No.</span><span>Dosage (wt%)</span><span>DH (%)</span><span>DP</span><span></span></div>';
  for(let i=1;i<=3;i++) pc+=`<div class="slot-row property-4"><span class="slot-index">${String(i).padStart(2,'0')}</span><input id="protective_colloid_pct${i}" type="number" min="0" step="0.001" placeholder="0.000"/><input id="protective_colloid_dh${i}" type="number" min="0" max="100" step="0.1" placeholder="DH"/><input id="protective_colloid_dp${i}" type="number" min="0" step="1" placeholder="DP"/><button class="clear-slot" data-clear="pc:${i}">×</button></div>`;
  $('protectiveColloidPropertySlots').innerHTML=pc;

  document.querySelectorAll('[data-clear]').forEach(btn=>btn.addEventListener('click',()=>{
    const [kind,i]=btn.dataset.clear.split(':');
    if(kind==='emulsifier'){['pct','eo'].forEach(k=>$( `emulsifier_${k}${i}` ).value='');}
    else {['pct','dh','dp'].forEach(k=>$( `protective_colloid_${k}${i}` ).value='');}
    updateMaterialCount();
  }));
  document.querySelectorAll('.property-slots input').forEach(x=>x.addEventListener('input',updateMaterialCount));
}

function renderAll(){Object.keys(families).forEach(renderFamily);renderPropertySlots();updateMaterialCount();}
function valueOrNull(id){const el=$(id);if(!el||el.value==='')return null;const v=Number(el.value);return Number.isFinite(v)?v:null;}

function updateMaterialCount(){
  let n=[...document.querySelectorAll('.material-name')].filter(x=>x.value.trim()).length;
  for(let i=1;i<=4;i++) if((valueOrNull(`emulsifier_pct${i}`)||0)>0)n++;
  for(let i=1;i<=3;i++) if((valueOrNull(`protective_colloid_pct${i}`)||0)>0)n++;
  const el=$('materialCount'); if(el) el.textContent=n;
}

function collectRaw(){
  const raw={init_temp:valueOrNull('init_temp'),poly_temp:valueOrNull('poly_temp'),poly_pressure:valueOrNull('poly_pressure'),time:valueOrNull('time'),initiator_feed_mode:$('initiator_feed_mode').value||null,coinitiator_pct:valueOrNull('coinitiator_pct'),dispersant_pct:valueOrNull('dispersant_pct'),pH_buffer_pct:valueOrNull('pH_buffer_pct')};
  Object.values(families).forEach(f=>{for(let i=1;i<=f.slots;i++){raw[f.typePrefix+i]=($(f.typePrefix+i)?.value||'').trim()||null;raw[f.pctPrefix+i]=valueOrNull(f.pctPrefix+i);}});
  for(let i=1;i<=4;i++){raw[`emulsifier_type${i}`]=null;raw[`emulsifier_pct${i}`]=valueOrNull(`emulsifier_pct${i}`);raw[`emulsifier_eo${i}`]=valueOrNull(`emulsifier_eo${i}`);}
  for(let i=1;i<=3;i++){raw[`protective_colloid_type${i}`]=null;raw[`protective_colloid_pct${i}`]=valueOrNull(`protective_colloid_pct${i}`);raw[`protective_colloid_dh${i}`]=valueOrNull(`protective_colloid_dh${i}`);raw[`protective_colloid_dp${i}`]=valueOrNull(`protective_colloid_dp${i}`);}
  return raw;
}

const processLabels={init_temp:'Initial Temperature',poly_temp:'Polymerization Temperature',poly_pressure:'Polymerization Pressure',time:'Polymerization Time',initiator_feed_mode:'Initiator Feeding Mode',coinitiator_pct:'Coinitiator Dosage',dispersant_pct:'Dispersant Dosage',pH_buffer_pct:'pH Buffer Dosage'};
function validateRaw(raw){
  const required=['init_temp','poly_temp','poly_pressure','time','initiator_feed_mode','coinitiator_pct','dispersant_pct','pH_buffer_pct'];
  const missing=required.filter(k=>raw[k]===null||raw[k]===''); const errors=[];
  Object.values(families).forEach(f=>{for(let i=1;i<=f.slots;i++){const n=raw[f.typePrefix+i],p=raw[f.pctPrefix+i];if(n&&(p===null||p<=0))errors.push(`${f.label} ${i}: dosage is required.`);if(!n&&p!==null&&p>0)errors.push(`${f.label} ${i}: select a material for the entered dosage.`);}});
  for(let i=1;i<=4;i++){const p=raw[`emulsifier_pct${i}`],eo=raw[`emulsifier_eo${i}`];if(p!==null&&p>0&&(eo===null||eo<0))errors.push(`Emulsifier ${i}: EO number is required.`);}
  for(let i=1;i<=3;i++){const p=raw[`protective_colloid_pct${i}`],dh=raw[`protective_colloid_dh${i}`],dp=raw[`protective_colloid_dp${i}`];if(p!==null&&p>0&&(dh===null||dp===null))errors.push(`Protective colloid ${i}: DH and DP are required.`);}
  return {missing,errors};
}

async function boot(){
  setupTabs();
  const params=new URLSearchParams(window.location.search);
  if(params.get('figure')==='1') document.body.classList.add('figure-mode');
  try{const r=await fetch('/api/raw-schema');if(r.ok)rawSchema=await r.json();}catch(e){}
  renderAll();
  const btn=$('predictBtn'),notice=$('assetNotice'),state=$('systemState'),badge=$('pipelineBadge'),badgeText=$('pipelineBadgeText');
  if(rawSchema.ready){
    state.textContent='Prediction pipeline ready · property-driven stabilizer input';state.className='ready';
    badge.className='pipeline-badge ready';badgeText.textContent='Prediction workflow ready · 4 property models';
    notice.classList.add('hidden');btn.disabled=false;
  } else {
    state.textContent=rawSchema.ui_ready?'UI ready · model bundle unavailable':'Required feature assets unavailable';state.className='error';
    badge.className='pipeline-badge error';badgeText.textContent=rawSchema.ui_ready?'Model bundle unavailable':'Feature assets unavailable';
    notice.classList.remove('hidden');
    const miss=(rawSchema.missing_assets||[]).join(', ');
    notice.innerHTML=`The interface is available, but live prediction requires the frozen four-model bundle and master feature reference.${miss?` Missing: ${escapeHtml(miss)}.`:''}`;
    btn.disabled=true;
  }
  const requestedTab=params.get('tab');
  if(requestedTab){const b=document.querySelector(`.tab[data-tab="${requestedTab}"]`); if(b) b.click();}
  if(params.get('figure')==='1' && rawSchema.ready){
    await loadExample();
    await predict();
  }
}
function clearAll(){document.querySelectorAll('input').forEach(x=>x.value='');document.querySelectorAll('.material-name').forEach(x=>x.value='');$('initiator_feed_mode').value='';$('coinitiator_pct').value='0';$('dispersant_pct').value='0';$('pH_buffer_pct').value='0';['vcValue','ecValue','scValue','hchoValue'].forEach(id=>$(id).textContent='—');$('statusBox').classList.add('hidden');$('warningBox').classList.add('hidden');const rs=$('resultState');rs.className='result-state waiting';rs.innerHTML='<span class="result-state-icon">○</span><div><strong>Awaiting prediction</strong><p>Load an example or enter a formulation, then run the frozen four-model workflow.</p></div>';updateMaterialCount();}
function setRawFormula(raw){['init_temp','poly_temp','poly_pressure','time','coinitiator_pct','dispersant_pct','pH_buffer_pct'].forEach(k=>{$(k).value=raw[k]??'';});$('initiator_feed_mode').value=raw.initiator_feed_mode??'';Object.values(families).forEach(f=>{for(let i=1;i<=f.slots;i++){$(f.typePrefix+i).value=raw[f.typePrefix+i]??'';$(f.pctPrefix+i).value=raw[f.pctPrefix+i]??'';}});for(let i=1;i<=4;i++){ $(`emulsifier_pct${i}`).value=raw[`emulsifier_pct${i}`]??''; $(`emulsifier_eo${i}`).value=raw[`emulsifier_eo${i}`]??'';}for(let i=1;i<=3;i++){ $(`protective_colloid_pct${i}`).value=raw[`protective_colloid_pct${i}`]??''; $(`protective_colloid_dh${i}`).value=raw[`protective_colloid_dh${i}`]??''; $(`protective_colloid_dp${i}`).value=raw[`protective_colloid_dp${i}`]??'';}updateMaterialCount();}
async function loadExample(){try{const r=await fetch('/api/raw-example');const data=await r.json();if(!r.ok)throw new Error(data.detail||'Failed to load example.');clearAll();setRawFormula(data.raw_formula);showStatus(`Example loaded from training-data row ${data.source_row}.`);}catch(e){showStatus(e.message,true);}}
function showStatus(text,isError=false){const b=$('statusBox');b.textContent=text;b.className='message status'+(isError?' error':'');b.classList.remove('hidden');}
function showWarnings(items){const b=$('warningBox');if(!items?.length){b.classList.add('hidden');b.innerHTML='';return;}b.classList.remove('hidden');b.innerHTML='<strong>Reference-range notice</strong><ul>'+items.map(x=>`<li>${escapeHtml(x)}</li>`).join('')+'</ul>';}
async function predict(){
  const raw=collectRaw(),v=validateRaw(raw);
  if(v.missing.length){showStatus('Complete all required process conditions: '+v.missing.map(k=>processLabels[k]||k).join(', '),true);return;}
  if(v.errors.length){showStatus(v.errors[0],true);return;}
  const btn=$('predictBtn'),rs=$('resultState');
  btn.disabled=true;btn.textContent='Building frozen features and predicting…';
  rs.className='result-state waiting';rs.innerHTML='<span class="result-state-icon">…</span><div><strong>Running frozen prediction workflow</strong><p>Constructing the master feature vector and evaluating four surrogate models.</p></div>';
  try{
    const r=await fetch('/api/raw-predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({raw_formula:raw})});
    const data=await r.json();if(!r.ok)throw new Error(data.detail||'Prediction failed.');
    const p=data.predictions;
    $('vcValue').textContent=Number(p.viscosity.value).toLocaleString('en-US',{maximumFractionDigits:0});
    $('ecValue').textContent=Number(p.ethylene_content.value).toFixed(2);
    $('scValue').textContent=Number(p.solid_content.value).toFixed(2);
    $('hchoValue').textContent=p.free_hcho.value<0.01?Number(p.free_hcho.value).toFixed(5):Number(p.free_hcho.value).toFixed(3);
    if(data.feature_engineering?.master_feature_count) $('featureCount').textContent=data.feature_engineering.master_feature_count;
    rs.className='result-state complete';
    rs.innerHTML='<span class="result-state-icon">✓</span><div><strong>Prediction completed successfully</strong><p>Frozen feature construction and all four property-specific surrogate-model evaluations completed.</p></div>';
    rs.dataset.predictionState='complete';
    showStatus('Prediction completed with property-driven EO/DH/DP inputs and the frozen feature pipeline.');showWarnings(data.warnings||[]);
  }catch(e){
    rs.className='result-state error';rs.innerHTML='<span class="result-state-icon">!</span><div><strong>Prediction could not be completed</strong><p>'+escapeHtml(e.message)+'</p></div>';showStatus(e.message,true);
  }finally{btn.disabled=!rawSchema.ready;btn.innerHTML='<span class="play-icon">▶</span> Predict Properties';}
}

$('clearBtn').addEventListener('click',clearAll);$('exampleBtn').addEventListener('click',loadExample);$('predictBtn').addEventListener('click',predict);boot().then(()=>{if($('coinitiator_pct').value==='')$('coinitiator_pct').value='0';if($('dispersant_pct').value==='')$('dispersant_pct').value='0';if($('pH_buffer_pct').value==='')$('pH_buffer_pct').value='0';});
