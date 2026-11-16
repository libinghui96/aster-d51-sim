// web/src/main.ts
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import loadMujoco from "mujoco-js";
import {
  AsterD51Thermo1D,
  CabControls,
  CylinderCalibration,
  SmokeboxLutJson,
  SpatialFluidState,
  StepOutput,
} from "./thermo1d";
import { AiDriverMode, HumanEngineerTS, IdealOracleTS } from "./aiDrivers";
import { BrassWhistleAudio, LiveSteamParticles } from "./particles";
import {
  FluidPartVisualizer,
  attachCopperBoilerShader,
  attachFeedwaterShader,
  attachSuperheaterShader,
} from "./fluidShaders";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const overlay = document.getElementById("loading-overlay")!;
const loadingText = document.getElementById("loading-text")!;
const partTooltip = document.getElementById("part-tooltip")!;

interface InteractivePartEntry {
  mesh: THREE.Mesh;
  mat: THREE.MeshStandardMaterial;
  geomName: string;
  label: string;
  kind: "drag" | "toggle" | "momentary" | "pump";
}

const INTERACTIVE_PART_META: Record<string, { label: string; kind: InteractivePartEntry["kind"] }> = {
  cab_regulator_vis: { label: "11-11 加減弁ハンドル (上下ドラッグで蒸気開度 11-10 扇形案内)", kind: "drag" },
  cab_reverser_wheel_vis: { label: "8-16 左運転台・扇形ノッチ式逆転機レバー (ドラッグで前後進カットオフ)", kind: "drag" },
  cab_blower_vis: { label: "11-13 左上3本スポーク通風弁手輪 → 13-8 外側通風管 (ドラッグで強制ドラフト)", kind: "drag" },
  cab_whistle_vis: { label: "11-3 左上外 S字汽笛レバー → 13-3 → 11-8 缶胴下真鍮汽笛 (押下ホールドで吹鳴)", kind: "momentary" },
  cab_sight_drain_vis: { label: "11-5 右側外縁(-Y) 11-7 水面計下部六角ドレン栓 (クリックでフラッシュ洗浄)", kind: "toggle" },
  cab_blowdown_vis: { label: "11-16/11-17 左中下マイナス溝ブロー弁 (ドラッグ/クリックで缶底排泥ブロー)", kind: "drag" },
  cab_fire_door_vis: { label: "12-1 低位中央・右ヒンジ火室扉 (ドラッグで開閉・二次空気流入)", kind: "drag" },
  cab_bypass_vis: { label: "6-9/8-7 右後床面・同軸給水バイパス弁コック (ドラッグで 31-14 戻水量調整)", kind: "drag" },
  cab_roof_hatch_vis: { label: "21-15/21-28 運転室屋根ベンチレーター天窓 (クリックで前ヒンジ開閉)", kind: "toggle" },
  drain_pull_rod_vis: { label: "8-6/8-5 右床下ドレン弁水平引き棒 (クリックで8段排液リンク開閉)", kind: "toggle" },
  tender_handpump_vis: { label: "31-3/31-5 炭水車後部水槽内・横型プランジャ手動ポンプ揺腕 (クリックで圧送)", kind: "pump" },
  tender_drain_plug_vis: { label: "31-11/31-9 水槽内・上引き排水スタンドパイプ栓 (クリックで炭水車底から放水)", kind: "toggle" },
  tender_coal_cover_vis: { label: "36-1/36-2 後部給水ハッチ折り蓋 (クリックで開閉・給水口/31-3/31-14 露出)", kind: "toggle" },
  tender_water_lid_vis: { label: "35-6 後端点検マンホール丸蓋 (クリックで開閉)", kind: "toggle" },
  tender_fuel_valve_vis: { label: "36-6 前部アルコール槽・左前ニードル弁 (ドラッグで 34-2 サンプ供油開度)", kind: "drag" },
  tender_fuel_cap_vis: { label: "36-5 前部アルコール槽・右気密燃料蓋 (クリックで開閉・開弁時マリオット負圧喪失)", kind: "toggle" },
};

function createPg20DialTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext("2d")!;
  const cx = 128, cy = 128, r = 112;
  ctx.fillStyle = "#f7f5ee";
  ctx.fillRect(0, 0, 256, 256);

  ctx.strokeStyle = "#22262c";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // 0..6 kgf/cm² 240° ブルドン管目盛り (-210° .. +30° 上基準時計回り, 4.0 kgf 赤線)
  const aStart = (-210 * Math.PI) / 180;
  const aSpan = (240 * Math.PI) / 180;
  const aRed = aStart + (4.0 / 6.0) * aSpan;
  ctx.strokeStyle = "#d92b2b";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(cx, cy, r - 15, aRed, aStart + aSpan);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "bold 24px ui-monospace, monospace";
  for (let i = 0; i <= 12; i++) {
    const pVal = i * 0.5;
    const ang = aStart + (pVal / 6.0) * aSpan;
    const isMajor = i % 2 === 0;
    const rOuter = r - 9;
    const rInner = isMajor ? r - 27 : r - 18;
    ctx.strokeStyle = pVal >= 4.0 ? "#d92b2b" : "#181b20";
    ctx.lineWidth = isMajor ? 4.2 : 2.2;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(ang) * rInner, cy + Math.sin(ang) * rInner);
    ctx.lineTo(cx + Math.cos(ang) * rOuter, cy + Math.sin(ang) * rOuter);
    ctx.stroke();
    if (isMajor) {
      ctx.fillStyle = pVal >= 4.0 ? "#c91e1e" : "#15181d";
      const rt = r - 45;
      ctx.fillText(String(pVal), cx + Math.cos(ang) * rt, cy + Math.sin(ang) * rt);
    }
  }
  ctx.fillStyle = "#333842";
  ctx.font = "bold 16px sans-serif";
  ctx.fillText("kgf/cm²", cx, cy + 44);
  ctx.fillText("PG-20", cx, cy + 66);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

async function main() {
  const mujoco = await loadMujoco();
  mujoco.FS.mkdir("/working");
  mujoco.FS.mkdir("/working/meshes");

  const modelBase = `${import.meta.env.BASE_URL}models`;
  const manifestResp = await fetch(`${modelBase}/manifest.json`);
  const manifest: { xmls: string[]; meshes: string[]; configs?: string[] } = await manifestResp.json();
  const stlLoader = new STLLoader();
  const stlGeomMap = new Map<string, THREE.BufferGeometry>();

  let smokeboxLut: SmokeboxLutJson | null = null;
  const cfgList = manifest.configs || [];
  if (cfgList.includes("smokebox_lut.json")) {
    try {
      smokeboxLut = await (await fetch(`${modelBase}/smokebox_lut.json`)).json();
    } catch {
      smokeboxLut = null;
    }
  }
  const humanAgent = new HumanEngineerTS();
  const oracleAgent = new IdealOracleTS();
  const whistleAudio = new BrassWhistleAudio();

  for (const xmlName of manifest.xmls) {
    const txt = await (await fetch(`${modelBase}/${xmlName}`)).text();
    mujoco.FS.writeFile(`/working/${xmlName}`, txt);
  }
  for (const stlName of manifest.meshes) {
    loadingText.textContent = `正在挂载 CAD 零件网格: ${stlName} ...`;
    const buf = await (await fetch(`${modelBase}/meshes/${stlName}`)).arrayBuffer();
    mujoco.FS.writeFile(`/working/meshes/${stlName}`, new Uint8Array(buf));
    const geom = stlLoader.parse(buf);
    geom.scale(0.001, 0.001, 0.001);
    geom.computeVertexNormals();
    stlGeomMap.set(stlName.replace(".stl", ""), geom);
  }

  const container = document.getElementById("canvas-container")!;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11151c);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.85;
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.005, 30.0);
  camera.position.set(0.28, -0.32, 0.17);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.target.set(0.02, 0.0, 0.065);

  scene.add(new THREE.HemisphereLight(0xddeeff, 0x2a3038, 1.15));
  const dirLight = new THREE.DirectionalLight(0xfff6e8, 2.4);
  dirLight.position.set(0.8, -1.2, 1.8);
  scene.add(dirLight);
  const rimLight = new THREE.DirectionalLight(0xc8dcff, 1.2);
  rimLight.position.set(-0.9, 1.1, 1.2);
  scene.add(rimLight);
  const fireboxGlowLight = new THREE.PointLight(0xff4214, 0.0, 0.35);
  scene.add(fireboxGlowLight);
  scene.add(new THREE.GridHelper(6, 60, 0x30363d, 0x1c2128).rotateX(Math.PI / 2));

  const steamParticles = new LiveSteamParticles();
  scene.add(steamParticles.points);

  const fluidVis = new FluidPartVisualizer();
  scene.add(fluidVis.root);
  scene.add(fluidVis.tenderRoot);

  let model: any = null;
  let data: any = null;
  let thermo: AsterD51Thermo1D;
  let currentCalib: CylinderCalibration;
  let geomMeshes: THREE.Mesh[] = [];
  let cylHousingMeshes: THREE.Mesh[] = [];
  let boilerOuterMeshes: THREE.Mesh[] = [];
  let copperCoreMeshes: THREE.Mesh[] = [];
  let lubeShellMeshes: THREE.Mesh[] = [];
  let tenderOuterMeshes: THREE.Mesh[] = [];
  let boilerShellMats: THREE.MeshStandardMaterial[] = [];
  let copperCoreMats: THREE.MeshStandardMaterial[] = [];
  let interactiveParts: InteractivePartEntry[] = [];
  let visJointMap: Record<string, [number, number]> = {};
  let handpumpVisTimer = 0.0;
  let handpumpVisPhase = 0.0;
  let simGroup = new THREE.Group();
  scene.add(simGroup);

  const ctrl: CabControls = {
    fuelValve: 1.0,
    burnerLit: true,
    chimneyFan: false,
    blowerValve: 0.25,
    regulator: 0.0,
    reverser: 0.85,
    drainCock: true,
    bypassValve: 0.40,
    whistleValve: 0.0,
    blowdownValve: 0.0,
    fireboxDoor: 0.0,
    fuelCapOpen: false,
    waterLidOpen: false,
    waterFillValve: 0.0,
    tenderDrainOpen: false,
    sightDrainValve: false,
    cabRoofOpen: false,
    coalCoverOpen: false,
    thermalWarp: 1.0,
    gradeDeg: 0.0,
  };
  let aiMode: AiDriverMode = "off";
  let aiStatusText = "";

  let idxMap: Record<string, number> = {};
  let isRollerMode = true;
  let tenderDryMass = 1.15;
  let prevFwdVel = 0.0;

  // -------------------------------------------------------------------------
  // 本地 NVIDIA RTX 5080 (571万体素真 3D 求解器) WebSocket 自动发现与热桥接
  // -------------------------------------------------------------------------
  let wsBridge: WebSocket | null = null;
  let wsConnected = false;
  let latestWsFrame: any = null;
  const urlParams = new URLSearchParams(window.location.search);
  const explicitWs = urlParams.get("ws");
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const hostName = window.location.hostname || "127.0.0.1";
  const wsCandidates: string[] = explicitWs
    ? [explicitWs]
    : import.meta.env.DEV
      ? [`ws://${hostName}:8765`, "ws://127.0.0.1:8765"]
      : [`${wsProto}//${window.location.host}/ws`, `ws://${hostName}:8765`, "ws://127.0.0.1:8765"];
  let wsCandidateIdx = 0;

  const sendWsMessage = (payload: Record<string, any>) => {
    if (wsBridge && wsConnected && wsBridge.readyState === WebSocket.OPEN) {
      wsBridge.send(JSON.stringify(payload));
    }
  };

  const pushControlsToWs = () => {
    sendWsMessage({ type: "ctrl", aiMode, autoFireman: aiMode !== "off", ctrl });
  };

  const triggerHandPumpBoth = (strokes = 5) => {
    thermo.triggerHandPump(strokes);
    handpumpVisTimer = Math.max(handpumpVisTimer, Math.min(3.0, 0.50 * Math.max(1, strokes)));
    sendWsMessage({ type: "action", name: "handpump", strokes });
  };

  const connectLocalGpuServer = () => {
    if (window.location.protocol === "https:" && !explicitWs) {
      return;
    }
    const targetUrl = wsCandidates[wsCandidateIdx % wsCandidates.length];
    let openedOnce = false;
    try {
      const sock = new WebSocket(targetUrl);
      wsBridge = sock;
      sock.onopen = () => {
        openedOnce = true;
        wsConnected = true;
        sendWsMessage({ type: "scene", scene: isRollerMode ? "roller" : "curve" });
        pushControlsToWs();
      };
      sock.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg && msg.type === "frame") {
            latestWsFrame = msg;
          }
        } catch {
          // ignore malformed frame
        }
      };
      sock.onclose = () => {
        wsConnected = false;
        latestWsFrame = null;
        if (!openedOnce) {
          wsCandidateIdx = (wsCandidateIdx + 1) % wsCandidates.length;
          window.setTimeout(connectLocalGpuServer, wsCandidateIdx === 0 ? 2500 : 120);
        } else {
          window.setTimeout(connectLocalGpuServer, 2200);
        }
      };
      sock.onerror = () => {
        sock.close();
      };
    } catch {
      wsConnected = false;
    }
  };

  function objId(objType: any, name: string): number {
    const t = typeof objType === "number" ? objType : objType.value;
    return mujoco.mj_name2id(model, t, name);
  }

  function injectRollerFlywheelInertia() {
    const q = new THREE.Quaternion();
    const axisY = new THREE.Vector3(0, 1, 0);
    const localAxis = new THREE.Vector3();
    for (let idx = 1; idx <= 4; idx++) {
      const bid = objId(mujoco.mjtObj.mjOBJ_BODY, `wheelset_${idx}`);
      if (bid < 0) continue;
      q.set(model.body_iquat[4 * bid + 1], model.body_iquat[4 * bid + 2], model.body_iquat[4 * bid + 3], model.body_iquat[4 * bid]);
      localAxis.copy(axisY).applyQuaternion(q.invert());
      const comps = [Math.abs(localAxis.x), Math.abs(localAxis.y), Math.abs(localAxis.z)];
      const maxAx = comps.indexOf(Math.max(...comps));
      for (let k = 0; k < 3; k++) {
        model.body_inertia[3 * bid + k] += k === maxAx ? 0.00045 : 0.00020;
      }
    }
  }

  function loadSimulationScene(xmlFile: string) {
    if (data) data.delete();
    if (model) model.delete();
    scene.remove(simGroup);
    simGroup = new THREE.Group();
    scene.add(simGroup);
    geomMeshes = [];
    cylHousingMeshes = [];
    boilerOuterMeshes = [];
    copperCoreMeshes = [];
    lubeShellMeshes = [];
    tenderOuterMeshes = [];
    boilerShellMats = [];
    copperCoreMats = [];
    interactiveParts = [];
    visJointMap = {};
    prevFwdVel = 0.0;

    model = mujoco.MjModel.loadFromXML(`/working/${xmlFile}`);
    isRollerMode = xmlFile === "d51_full.xml";
    if (isRollerMode) {
      injectRollerFlywheelInertia();
    }

    const jRev = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_reverser");
    const actRev = objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl");
    model.jnt_range[2 * jRev] = -0.48;
    model.jnt_range[2 * jRev + 1] = 0.48;
    model.actuator_ctrlrange[2 * actRev] = -0.46;
    model.actuator_ctrlrange[2 * actRev + 1] = 0.46;
    model.actuator_gainprm[10 * actRev] = 60.0;
    model.actuator_biasprm[10 * actRev + 1] = -60.0;

    const jPL = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_L");
    const jPR = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_R");
    const jValL = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_valve_L");
    const jValR = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_valve_R");
    const jW3 = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_wheel_3");
    const jPump = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_pump_ram");
    const jDrain = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_drain_cock");

    const sAdr = (sName: string) => model.sensor_adr[objId(mujoco.mjtObj.mjOBJ_SENSOR, sName)];
    idxMap = {
      sPL: sAdr("piston_pos_L"), sVL: sAdr("piston_vel_L"), sValL: sAdr("valve_pos_L"),
      sPR: sAdr("piston_pos_R"), sVR: sAdr("piston_vel_R"), sValR: sAdr("valve_pos_R"),
      sW3: sAdr("wheel3_omega"), sPump: sAdr("pump_ram_vel"),
      actFL: objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "steam_force_L"),
      actFR: objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "steam_force_R"),
      actRev,
      qvelW3: model.jnt_dofadr[jW3],
      qvelPump: model.jnt_dofadr[jPump],
      qvelValL: model.jnt_dofadr[jValL],
      qvelValR: model.jnt_dofadr[jValR],
      qvelDrain: jDrain >= 0 ? model.jnt_dofadr[jDrain] : -1,
      locoBody: objId(mujoco.mjtObj.mjOBJ_BODY, "loco_frame"),
      tenderBody: objId(mujoco.mjtObj.mjOBJ_BODY, "tender_body"),
    };

    for (const jn of [
      "j_cab_regulator",
      "j_cab_blower",
      "j_cab_whistle",
      "j_cab_sight_drain",
      "j_cab_pg_needle",
      "j_cab_blowdown",
      "j_cab_fire_door",
      "j_cab_bypass",
      "j_cab_roof_hatch",
      "j_tender_handpump",
      "j_tender_hp_ram",
      "j_tender_drain_plug",
      "j_tender_coal_cover",
      "j_tender_water_lid",
      "j_tender_fuel_valve",
      "j_tender_fuel_cap",
    ]) {
      const jid = objId(mujoco.mjtObj.mjOBJ_JOINT, jn);
      if (jid >= 0) {
        visJointMap[jn] = [model.jnt_qposadr[jid], model.jnt_dofadr[jid]];
      }
    }

    if (idxMap.tenderBody >= 0) {
      tenderDryMass = Math.max(0.6, model.body_mass[idxMap.tenderBody] - 0.250);
    }

    const savedDisable = model.opt.disableflags;
    const bitVal = (b: any) => (typeof b === "number" ? b : b.value);
    model.opt.disableflags =
      savedDisable |
      bitVal(mujoco.mjtDisableBit.mjDSBL_CONTACT) |
      bitVal(mujoco.mjtDisableBit.mjDSBL_GRAVITY);
    const calData = new mujoco.MjData(model);
    calData.ctrl[idxMap.actRev] = 0.0;
    let vMinL = 1e9, vMaxL = -1e9, vMinR = 1e9, vMaxR = -1e9;
    for (let i = 0; i < 5000; i++) {
      if (!isRollerMode) for (let k = 0; k < 6; k++) calData.qvel[k] = 0;
      calData.qfrc_applied[idxMap.qvelW3] = 0.04 * (2.0 * Math.PI - calData.qvel[idxMap.qvelW3]);
      mujoco.mj_step(model, calData);
      if (i >= 1500) {
        const vl = calData.sensordata[idxMap.sValL];
        const vr = calData.sensordata[idxMap.sValR];
        vMinL = Math.min(vMinL, vl); vMaxL = Math.max(vMaxL, vl);
        vMinR = Math.min(vMinR, vr); vMaxR = Math.max(vMaxR, vr);
      }
    }
    calData.delete();
    model.opt.disableflags = savedDisable;

    const leadAmp = 0.5 * (vMaxL - vMinL);
    currentCalib = {
      qFdcL: model.jnt_range[2 * jPL + 1] - 0.002,
      qFdcR: model.jnt_range[2 * jPR + 1] - 0.002,
      stroke: model.jnt_range[2 * jPL + 1] - model.jnt_range[2 * jPL] - 0.004,
      qValveCenterL: 0.5 * (vMaxL + vMinL),
      qValveCenterR: 0.5 * (vMaxR + vMinR),
      steamLap: 0.99 * leadAmp,
      exhaustLap: 0.02 * leadAmp,
      portWidth: 0.95 * leadAmp,
    };

    data = new mujoco.MjData(model);
    mujoco.mj_forward(model, data);
    thermo = new AsterD51Thermo1D(currentCalib, smokeboxLut);

    const matIron = new THREE.MeshStandardMaterial({ color: 0x282c34, metalness: 0.58, roughness: 0.38 });
    const matSteel = new THREE.MeshStandardMaterial({ color: 0xd8dee9, metalness: 0.88, roughness: 0.22 });
    const matBrass = new THREE.MeshStandardMaterial({ color: 0xd8a03c, metalness: 0.82, roughness: 0.26 });
    const matCopper = new THREE.MeshStandardMaterial({ color: 0xc96f46, metalness: 0.86, roughness: 0.24 });
    const matSiliconeHose = new THREE.MeshStandardMaterial({
      color: 0xf4f1e6,
      metalness: 0.04,
      roughness: 0.36,
      transparent: true,
      opacity: 0.86,
    });
    const matVitonHose = new THREE.MeshStandardMaterial({
      color: 0x1a1d22,
      metalness: 0.08,
      roughness: 0.78,
    });
    const matPyrexGlass = new THREE.MeshStandardMaterial({
      color: 0xd9f7ff,
      metalness: 0.06,
      roughness: 0.06,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    });
    const matGaugeDial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: createPg20DialTexture(),
      metalness: 0.04,
      roughness: 0.32,
    });

    // 挂载零件级沿程相变与给水脉冲 PBR 着色器实例
    const matSuperheater = matSteel.clone();
    attachSuperheaterShader(matSuperheater, fluidVis.uniforms);
    const matFeedwater = matBrass.clone();
    attachFeedwaterShader(matFeedwater, fluidVis.uniforms, false);
    const matWaterHose = matVitonHose.clone();
    attachFeedwaterShader(matWaterHose, fluidVis.uniforms, true);

    const pbrMatByName: Record<string, THREE.MeshStandardMaterial> = {
      mat_black_iron: matIron,
      mat_machined_steel: matSteel,
      mat_brass: matBrass,
      mat_copper: matCopper,
      mat_silicone_hose: matSiliconeHose,
      mat_viton_hose: matVitonHose,
      mat_pyrex_glass: matPyrexGlass,
      mat_gauge_dial: matGaugeDial,
    };

    const xmlDoc = new DOMParser().parseFromString(
      new TextDecoder().decode(mujoco.FS.readFile(`/working/${xmlFile}`)), "text/xml"
    );
    const meshTags = Array.from(xmlDoc.querySelectorAll("asset > mesh"));
    const meshIdToKey = meshTags.map((el) => (el.getAttribute("file") || "").replace(".stl", ""));
    const matTags = Array.from(xmlDoc.querySelectorAll("asset > material"));
    const matIdToName = matTags.map((el) => el.getAttribute("name") || "");
    const geomTags = Array.from(xmlDoc.querySelectorAll("worldbody geom"));
    const geomNameToMat = new Map<string, string>();
    for (const gel of geomTags) {
      const gn = gel.getAttribute("name");
      const gm = gel.getAttribute("material");
      if (gn && gm) geomNameToMat.set(gn, gm);
    }

    const alignedMeshCache = new Map<number, THREE.BufferGeometry>();
    const meshObjType = typeof mujoco.mjtObj.mjOBJ_MESH === "number" ? mujoco.mjtObj.mjOBJ_MESH : mujoco.mjtObj.mjOBJ_MESH.value;
    const matObjType = typeof mujoco.mjtObj.mjOBJ_MATERIAL === "number" ? mujoco.mjtObj.mjOBJ_MATERIAL : mujoco.mjtObj.mjOBJ_MATERIAL.value;
    const geomObjType = typeof mujoco.mjtObj.mjOBJ_GEOM === "number" ? mujoco.mjtObj.mjOBJ_GEOM : mujoco.mjtObj.mjOBJ_GEOM.value;

    const resolveMjcfMatName = (gIdx: number, gName: string): string => {
      const matId = model.geom_matid ? model.geom_matid[gIdx] : -1;
      if (matId >= 0) {
        const byApi = mujoco.mj_id2name(model, matObjType, matId) || "";
        if (byApi) return byApi;
        if (matIdToName[matId]) return matIdToName[matId];
      }
      return geomNameToMat.get(gName) || "";
    };

    for (let g = 0; g < model.ngeom; g++) {
      const grp = model.geom_group[g];
      const rgbaA = model.geom_rgba[4 * g + 3];
      const gType = model.geom_type[g];
      const gName: string = mujoco.mj_id2name(model, geomObjType, g) || "";
      const mjMatName = resolveMjcfMatName(g, gName);
      let threeGeom: THREE.BufferGeometry | null = null;
      let mat: THREE.Material = pbrMatByName[mjMatName] || matIron;
      let meshKey = "";

      if (gType === 7) {
        const dataId = model.geom_dataid[g];
        const mjMeshName: string = mujoco.mj_id2name(model, meshObjType, dataId) || "";
        meshKey = mjMeshName ? mjMeshName.replace(/^m_/, "") : (meshIdToKey[dataId] || "");

        if (alignedMeshCache.has(dataId)) {
          threeGeom = alignedMeshCache.get(dataId)!;
        } else {
          const rawGeom = stlGeomMap.get(meshKey);
          if (rawGeom) {
            const aligned = rawGeom.clone();
            const posAttr = rawGeom.getAttribute("position");
            aligned.setAttribute("aCadPos", posAttr.clone());
            if (meshKey === "cab_pg_dial") {
              // 将 256×256 PG-ø20 白底刻度盘平面投影至 +22° 倾斜表盘面 (法向 [-cos22°, 0, +sin22°])
              const uvArr = new Float32Array(posAttr.count * 2);
              for (let vi = 0; vi < posAttr.count; vi++) {
                const px = posAttr.getX(vi);
                const py = posAttr.getY(vi);
                const pz = posAttr.getZ(vi);
                const u = 0.5 - py / 0.0168;
                const v = 0.5 + (0.374607 * px + 0.927184 * pz) / 0.0168;
                uvArr[2 * vi] = Math.max(0.02, Math.min(0.98, u));
                uvArr[2 * vi + 1] = Math.max(0.02, Math.min(0.98, v));
              }
              aligned.setAttribute("uv", new THREE.BufferAttribute(uvArr, 2));
            }
            aligned.translate(
              -model.mesh_pos[3 * dataId],
              -model.mesh_pos[3 * dataId + 1],
              -model.mesh_pos[3 * dataId + 2]
            );
            const qInv = new THREE.Quaternion(
              model.mesh_quat[4 * dataId + 1],
              model.mesh_quat[4 * dataId + 2],
              model.mesh_quat[4 * dataId + 3],
              model.mesh_quat[4 * dataId]
            ).invert();
            aligned.applyQuaternion(qInv);
            aligned.computeVertexNormals();
            alignedMeshCache.set(dataId, aligned);
            threeGeom = aligned;
          }
        }

        const baseMat = pbrMatByName[mjMatName] || matIron;
        if (meshKey === "boiler_firebox_flues") {
          const cuMat = baseMat.clone();
          attachCopperBoilerShader(cuMat, fluidVis.uniforms);
          copperCoreMats.push(cuMat);
          mat = cuMat;
        } else if (meshKey === "superheater_piping") {
          mat = matSuperheater;
        } else if (meshKey === "feedwater_piping") {
          mat = matFeedwater;
        } else if (gName === "umb_hose_feed" || gName === "umb_hose_return") {
          mat = matWaterHose;
        } else if (gName === "umb_hose_fuel") {
          mat = matSiliconeHose;
        } else if (INTERACTIVE_PART_META[gName]) {
          mat = baseMat.clone();
        } else if (meshKey === "displacement_lubricator") {
          mat = baseMat.clone();
        } else if (meshKey === "boiler_superstructure") {
          const bMat = baseMat.clone();
          boilerShellMats.push(bMat);
          mat = bMat;
        } else if (meshKey === "cylinder_block_L" || meshKey === "cylinder_block_R" || meshKey === "tender_body") {
          mat = baseMat.clone();
        } else {
          mat = baseMat;
        }
      } else if (grp < 3 && rgbaA > 0.05 && gType !== 0) {
        const sx = model.geom_size[3 * g];
        const sy = model.geom_size[3 * g + 1];
        const sz = model.geom_size[3 * g + 2];
        if (gType === 6) {
          threeGeom = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
        } else if (gType === 5) {
          threeGeom = new THREE.CylinderGeometry(sx, sx, 2 * sy, 20);
          threeGeom.rotateX(Math.PI / 2);
        } else if (gType === 3) {
          threeGeom = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16);
          threeGeom.rotateX(Math.PI / 2);
        }
        if (mjMatName && pbrMatByName[mjMatName]) {
          mat = pbrMatByName[mjMatName];
        } else {
          const col = new THREE.Color(model.geom_rgba[4 * g], model.geom_rgba[4 * g + 1], model.geom_rgba[4 * g + 2]);
          mat = new THREE.MeshStandardMaterial({ color: col, metalness: 0.55, roughness: 0.42 });
        }
      }

      const mesh = new THREE.Mesh(threeGeom || new THREE.BufferGeometry(), mat);
      mesh.name = gName;
      mesh.visible = threeGeom !== null && grp < 3 && rgbaA > 0.05;
      if (threeGeom && gType === 7) {
        if (meshKey === "cylinder_block_L" || meshKey === "cylinder_block_R") {
          cylHousingMeshes.push(mesh);
        } else if (meshKey === "boiler_superstructure") {
          boilerOuterMeshes.push(mesh);
        } else if (meshKey === "boiler_firebox_flues") {
          copperCoreMeshes.push(mesh);
        } else if (meshKey === "displacement_lubricator") {
          lubeShellMeshes.push(mesh);
        } else if (meshKey === "tender_body") {
          tenderOuterMeshes.push(mesh);
        }
        const meta = INTERACTIVE_PART_META[gName];
        if (meta && mat instanceof THREE.MeshStandardMaterial) {
          interactiveParts.push({
            mesh,
            mat,
            geomName: gName,
            label: meta.label,
            kind: meta.kind,
          });
        }
      }
      simGroup.add(mesh);
      geomMeshes.push(mesh);
    }
  }

  loadSimulationScene("d51_full.xml");
  connectLocalGpuServer();
  overlay.style.display = "none";

  const elReg = document.getElementById("rng-regulator") as HTMLInputElement;
  const elRev = document.getElementById("rng-reverser") as HTMLInputElement;
  const elBlw = document.getElementById("rng-blower") as HTMLInputElement;
  const elByp = document.getElementById("rng-bypass") as HTMLInputElement;
  const elFuel = document.getElementById("rng-fuel") as HTMLInputElement;
  const elWaterFill = document.getElementById("rng-waterfill") as HTMLInputElement;
  const elGrade = document.getElementById("rng-grade") as HTMLInputElement;
  const selAiMode = document.getElementById("sel-ai-mode") as HTMLSelectElement;
  const elAiStatus = document.getElementById("txt-ai-status")!;
  const btnBurner = document.getElementById("btn-burner")!;
  const btnFan = document.getElementById("btn-fan")!;
  const btnDrain = document.getElementById("btn-drain")!;
  const btnWhistle = document.getElementById("btn-whistle")!;
  const btnSightDrain = document.getElementById("btn-sightdrain")!;
  const btnFireDoor = document.getElementById("btn-firedoor")!;
  const btnBlowdown = document.getElementById("btn-blowdown")!;
  const btnRoofHatch = document.getElementById("btn-roofhatch")!;
  const btnTenderDrain = document.getElementById("btn-tenderdrain")!;
  const btnCoalCover = document.getElementById("btn-coalcover")!;
  const btnWaterLid = document.getElementById("btn-waterlid")!;
  const btnFuelCap = document.getElementById("btn-fuelcap")!;
  const btnWarp = document.getElementById("btn-warp")!;
  const btnHotPreset = document.getElementById("btn-hot-preset")!;
  const btnColdReset = document.getElementById("btn-cold-reset")!;

  const syncDomFromControls = () => {
    elReg.value = ctrl.regulator.toFixed(2);
    document.getElementById("val-regulator")!.textContent = `${Math.round(ctrl.regulator * 100)}%`;
    elRev.value = ctrl.reverser.toFixed(2);
    document.getElementById("val-reverser")!.textContent = `${ctrl.reverser >= 0 ? "+" : ""}${Math.round(ctrl.reverser * 100)}%`;
    elBlw.value = ctrl.blowerValve.toFixed(2);
    document.getElementById("val-blower")!.textContent = `${Math.round(ctrl.blowerValve * 100)}%`;
    elByp.value = ctrl.bypassValve.toFixed(2);
    document.getElementById("val-bypass")!.textContent = `${Math.round(ctrl.bypassValve * 100)}%`;
    elFuel.value = ctrl.fuelValve.toFixed(2);
    document.getElementById("val-fuel")!.textContent = `${Math.round(ctrl.fuelValve * 100)}%`;
    elWaterFill.value = ctrl.waterFillValve.toFixed(2);
    document.getElementById("val-waterfill")!.textContent = `${Math.round(ctrl.waterFillValve * 100)}%`;
    elGrade.value = ctrl.gradeDeg.toFixed(1);
    document.getElementById("val-grade")!.textContent = `${ctrl.gradeDeg >= 0 ? "+" : ""}${ctrl.gradeDeg.toFixed(1)}°`;

    btnBurner.classList.toggle("active", ctrl.burnerLit);
    btnFan.classList.toggle("active", ctrl.chimneyFan);
    btnDrain.classList.toggle("active", ctrl.drainCock);
    btnWhistle.classList.toggle("active", ctrl.whistleValve > 0.1);
    btnSightDrain.classList.toggle("active", ctrl.sightDrainValve);
    btnFireDoor.classList.toggle("active", ctrl.fireboxDoor > 0.1);
    btnBlowdown.classList.toggle("active", ctrl.blowdownValve > 0.1);
    btnRoofHatch.classList.toggle("active", ctrl.cabRoofOpen);
    btnTenderDrain.classList.toggle("active", ctrl.tenderDrainOpen);
    btnCoalCover.classList.toggle("active", ctrl.coalCoverOpen);
    btnWaterLid.classList.toggle("active", ctrl.waterLidOpen);
    btnFuelCap.classList.toggle("danger-active", ctrl.fuelCapOpen);
    btnWarp.classList.toggle("active", ctrl.thermalWarp > 2.0);
    selAiMode.value = aiMode;
    selAiMode.classList.toggle("ai-select-active", aiMode !== "off");
  };

  const setAiModeLocal = (newMode: AiDriverMode, notifyWs = true) => {
    aiMode = newMode;
    const dir = ctrl.reverser >= 0 ? 1.0 : -1.0;
    if (aiMode === "human") {
      humanAgent.reset(dir);
      aiStatusText = humanAgent.statusSummary;
    } else if (aiMode === "oracle") {
      oracleAgent.reset(dir);
      aiStatusText = oracleAgent.statusSummary;
    } else {
      aiStatusText = "";
    }
    syncDomFromControls();
    if (notifyWs) pushControlsToWs();
  };

  selAiMode.addEventListener("change", () => {
    setAiModeLocal(selAiMode.value as AiDriverMode, true);
  });

  const bindRange = (id: string, valId: string, fmt: (v: number) => string, cb: (v: number) => void, manualOverride = true) => {
    const el = document.getElementById(id) as HTMLInputElement;
    const txt = document.getElementById(valId)!;
    el.addEventListener("input", () => {
      if (manualOverride && aiMode !== "off") {
        setAiModeLocal("off", false);
      }
      const v = parseFloat(el.value);
      txt.textContent = fmt(v);
      cb(v);
      pushControlsToWs();
    });
  };
  bindRange("rng-regulator", "val-regulator", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.regulator = v));
  bindRange("rng-reverser", "val-reverser", (v) => `${v >= 0 ? "+" : ""}${Math.round(v * 100)}%`, (v) => (ctrl.reverser = v));
  bindRange("rng-blower", "val-blower", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.blowerValve = v));
  bindRange("rng-bypass", "val-bypass", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.bypassValve = v));
  bindRange("rng-fuel", "val-fuel", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.fuelValve = v));
  bindRange("rng-waterfill", "val-waterfill", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.waterFillValve = v), false);
  bindRange("rng-grade", "val-grade", (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}°`, (v) => (ctrl.gradeDeg = v), false);

  btnBurner.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.burnerLit = !ctrl.burnerLit;
    if (ctrl.burnerLit && ctrl.fuelValve < 0.25) ctrl.fuelValve = 0.85;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnFan.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.chimneyFan = !ctrl.chimneyFan;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnDrain.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.drainCock = !ctrl.drainCock;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnFireDoor.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.fireboxDoor = ctrl.fireboxDoor > 0.1 ? 0.0 : 1.0;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnBlowdown.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.blowdownValve = ctrl.blowdownValve > 0.1 ? 0.0 : 0.85;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnSightDrain.addEventListener("click", () => {
    ctrl.sightDrainValve = !ctrl.sightDrainValve;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnRoofHatch.addEventListener("click", () => {
    ctrl.cabRoofOpen = !ctrl.cabRoofOpen;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnTenderDrain.addEventListener("click", () => {
    ctrl.tenderDrainOpen = !ctrl.tenderDrainOpen;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnCoalCover.addEventListener("click", () => {
    ctrl.coalCoverOpen = !ctrl.coalCoverOpen;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnWaterLid.addEventListener("click", () => {
    ctrl.waterLidOpen = !ctrl.waterLidOpen;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnFuelCap.addEventListener("click", () => {
    if (aiMode !== "off") setAiModeLocal("off", false);
    ctrl.fuelCapOpen = !ctrl.fuelCapOpen;
    syncDomFromControls();
    pushControlsToWs();
  });
  btnWarp.addEventListener("click", () => {
    ctrl.thermalWarp = ctrl.thermalWarp > 2 ? 1.0 : 15.0;
    syncDomFromControls();
    pushControlsToWs();
  });

  // 11-3 弹簧复位汽笛手柄 (按住吹鸣，松开复位)
  const setWhistleState = (open: boolean) => {
    ctrl.whistleValve = open ? 1.0 : 0.0;
    syncDomFromControls();
    pushControlsToWs();
  };
  btnWhistle.addEventListener("pointerdown", () => setWhistleState(true));
  btnWhistle.addEventListener("pointerup", () => setWhistleState(false));
  btnWhistle.addEventListener("pointerleave", () => {
    if (ctrl.whistleValve > 0) setWhistleState(false);
  });

  document.getElementById("btn-handpump")!.addEventListener("click", () => {
    triggerHandPumpBoth(5);
  });

  btnHotPreset.addEventListener("click", () => {
    thermo.resetHotReady();
    Object.assign(ctrl, {
      fuelValve: 1.0,
      burnerLit: true,
      chimneyFan: false,
      blowerValve: 0.25,
      regulator: 0.0,
      reverser: 0.85,
      drainCock: true,
      bypassValve: 0.40,
      whistleValve: 0.0,
      blowdownValve: 0.0,
      fireboxDoor: 0.0,
      fuelCapOpen: false,
      waterLidOpen: false,
      waterFillValve: 0.0,
      tenderDrainOpen: false,
      sightDrainValve: false,
      cabRoofOpen: false,
      coalCoverOpen: false,
      thermalWarp: 1.0,
    });
    btnHotPreset.classList.add("active");
    btnColdReset.classList.remove("active");
    syncDomFromControls();
    sendWsMessage({ type: "action", name: "hotPreset" });
    pushControlsToWs();
  });

  btnColdReset.addEventListener("click", () => {
    mujoco.mj_resetData(model, data);
    mujoco.mj_forward(model, data);
    thermo.resetColdPrepared();
    Object.assign(ctrl, {
      fuelValve: 0.0,
      burnerLit: false,
      chimneyFan: false,
      blowerValve: 0.0,
      regulator: 0.0,
      reverser: 0.0,
      drainCock: true,
      bypassValve: 0.50,
      whistleValve: 0.0,
      blowdownValve: 0.0,
      fireboxDoor: 0.0,
      fuelCapOpen: false,
      waterLidOpen: false,
      waterFillValve: 0.0,
      tenderDrainOpen: false,
      sightDrainValve: false,
      cabRoofOpen: false,
      coalCoverOpen: false,
      thermalWarp: 1.0,
    });
    setAiModeLocal("off", false);
    btnColdReset.classList.add("active");
    btnHotPreset.classList.remove("active");
    syncDomFromControls();
    sendWsMessage({ type: "action", name: "coldReset" });
  });

  // -------------------------------------------------------------------------
  // P2-E / P2-F: 3D 视口鼠标 Raycaster 直接点击/拖拽驾驶室与煤水车阀柄
  // -------------------------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  let hoveredPart: InteractivePartEntry | null = null;
  let activeDragPart: InteractivePartEntry | null = null;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartVal = 0;

  const formatPartValue = (gName: string): string => {
    switch (gName) {
      case "cab_regulator_vis": return `${Math.round(ctrl.regulator * 100)}%`;
      case "cab_reverser_wheel_vis": return `${ctrl.reverser >= 0 ? "+" : ""}${Math.round(ctrl.reverser * 100)}%`;
      case "cab_blower_vis": return `${Math.round(ctrl.blowerValve * 100)}%`;
      case "cab_whistle_vis": return ctrl.whistleValve > 0.1 ? "吹鳴中" : "閉";
      case "cab_sight_drain_vis": return ctrl.sightDrainValve ? "吹抜中" : "閉";
      case "cab_blowdown_vis": return `${Math.round(ctrl.blowdownValve * 100)}%`;
      case "cab_fire_door_vis": return `${Math.round(ctrl.fireboxDoor * 100)}% 開`;
      case "cab_bypass_vis": return `${Math.round(ctrl.bypassValve * 100)}%`;
      case "cab_roof_hatch_vis": return ctrl.cabRoofOpen ? "開" : "閉";
      case "drain_pull_rod_vis": return ctrl.drainCock ? "開 (排液)" : "閉";
      case "tender_handpump_vis": return "クリック圧送";
      case "tender_drain_plug_vis": return ctrl.tenderDrainOpen ? "放水中" : "閉";
      case "tender_coal_cover_vis": return ctrl.coalCoverOpen ? "開" : "閉";
      case "tender_water_lid_vis": return ctrl.waterLidOpen ? "開" : "閉";
      case "tender_fuel_valve_vis": return `${Math.round(ctrl.fuelValve * 100)}%`;
      case "tender_fuel_cap_vis": return ctrl.fuelCapOpen ? "開 (大気開放)" : "気密閉";
      default: return "";
    }
  };

  const setHoveredPart = (entry: InteractivePartEntry | null, clientX = 0, clientY = 0) => {
    if (hoveredPart && hoveredPart !== entry) {
      hoveredPart.mat.emissive.setRGB(0, 0, 0);
    }
    hoveredPart = entry;
    if (hoveredPart) {
      hoveredPart.mat.emissive.setRGB(0.42, 0.24, 0.04);
      renderer.domElement.style.cursor = hoveredPart.kind === "drag" ? "ns-resize" : "pointer";
      partTooltip.style.display = "block";
      partTooltip.style.left = `${Math.min(window.innerWidth - 310, clientX + 14)}px`;
      partTooltip.style.top = `${Math.max(12, clientY - 30)}px`;
      partTooltip.textContent = `${hoveredPart.label} [${formatPartValue(hoveredPart.geomName)}]`;
    } else {
      renderer.domElement.style.cursor = "default";
      partTooltip.style.display = "none";
    }
  };

  const pickInteractivePart = (ev: PointerEvent): InteractivePartEntry | null => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointerNdc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointerNdc, camera);
    const meshes = interactiveParts.map((p) => p.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length === 0) return null;
    return interactiveParts.find((p) => p.mesh === hits[0].object) || null;
  };

  renderer.domElement.addEventListener("pointermove", (ev) => {
    if (activeDragPart) {
      const dx = ev.clientX - dragStartX;
      const dy = dragStartY - ev.clientY; // 向上拖动为正
      const delta = (dy + 0.45 * dx) / 135.0;
      const gn = activeDragPart.geomName;
      if (gn === "cab_regulator_vis") ctrl.regulator = Math.max(0, Math.min(1, dragStartVal + delta));
      else if (gn === "cab_reverser_wheel_vis") ctrl.reverser = Math.max(-1, Math.min(1, dragStartVal + delta * 1.4));
      else if (gn === "cab_blower_vis") ctrl.blowerValve = Math.max(0, Math.min(1, dragStartVal + delta));
      else if (gn === "cab_bypass_vis") ctrl.bypassValve = Math.max(0, Math.min(1, dragStartVal + delta));
      else if (gn === "cab_blowdown_vis") ctrl.blowdownValve = Math.max(0, Math.min(1, dragStartVal + delta));
      else if (gn === "tender_fuel_valve_vis") ctrl.fuelValve = Math.max(0, Math.min(1, dragStartVal + delta));
      else if (gn === "cab_fire_door_vis") ctrl.fireboxDoor = Math.max(0, Math.min(1, dragStartVal + delta));
      syncDomFromControls();
      pushControlsToWs();
      partTooltip.style.left = `${Math.min(window.innerWidth - 310, ev.clientX + 14)}px`;
      partTooltip.style.top = `${Math.max(12, ev.clientY - 30)}px`;
      partTooltip.textContent = `${activeDragPart.label} [${formatPartValue(gn)}]`;
      return;
    }
    const hit = pickInteractivePart(ev);
    setHoveredPart(hit, ev.clientX, ev.clientY);
  });

  renderer.domElement.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const hit = pickInteractivePart(ev);
    if (!hit) return;
    orbit.enabled = false;
    const gn = hit.geomName;
    if (hit.kind === "drag") {
      if (aiMode !== "off") setAiModeLocal("off", false);
      activeDragPart = hit;
      dragStartX = ev.clientX;
      dragStartY = ev.clientY;
      dragStartVal =
        gn === "cab_regulator_vis" ? ctrl.regulator :
        gn === "cab_reverser_wheel_vis" ? ctrl.reverser :
        gn === "cab_blower_vis" ? ctrl.blowerValve :
        gn === "cab_bypass_vis" ? ctrl.bypassValve :
        gn === "cab_blowdown_vis" ? ctrl.blowdownValve :
        gn === "tender_fuel_valve_vis" ? ctrl.fuelValve :
        ctrl.fireboxDoor;
    } else if (hit.kind === "momentary") {
      activeDragPart = hit;
      setWhistleState(true);
    } else if (hit.kind === "pump") {
      triggerHandPumpBoth(2);
    } else if (hit.kind === "toggle") {
      if (gn === "drain_pull_rod_vis") {
        if (aiMode !== "off") setAiModeLocal("off", false);
        ctrl.drainCock = !ctrl.drainCock;
      } else if (gn === "cab_sight_drain_vis") {
        ctrl.sightDrainValve = !ctrl.sightDrainValve;
      } else if (gn === "cab_roof_hatch_vis") {
        ctrl.cabRoofOpen = !ctrl.cabRoofOpen;
      } else if (gn === "tender_drain_plug_vis") {
        ctrl.tenderDrainOpen = !ctrl.tenderDrainOpen;
      } else if (gn === "tender_coal_cover_vis") {
        ctrl.coalCoverOpen = !ctrl.coalCoverOpen;
      } else if (gn === "tender_water_lid_vis") {
        ctrl.waterLidOpen = !ctrl.waterLidOpen;
      } else if (gn === "tender_fuel_cap_vis") {
        if (aiMode !== "off") setAiModeLocal("off", false);
        ctrl.fuelCapOpen = !ctrl.fuelCapOpen;
      }
      syncDomFromControls();
      pushControlsToWs();
    }
    setHoveredPart(hit, ev.clientX, ev.clientY);
  });

  const endPointerDrag = () => {
    if (activeDragPart?.kind === "momentary") {
      setWhistleState(false);
    }
    activeDragPart = null;
    orbit.enabled = true;
  };
  window.addEventListener("pointerup", endPointerDrag);
  window.addEventListener("pointercancel", endPointerDrag);

  const selScene = document.getElementById("select-scene") as HTMLSelectElement;
  const selCam = document.getElementById("select-cam") as HTMLSelectElement;

  const setXrayMatGroup = (meshes: THREE.Mesh[], xray: boolean, alpha: number, order: number) => {
    for (const m of meshes) {
      const mat = m.material as THREE.MeshStandardMaterial;
      mat.transparent = xray;
      mat.opacity = xray ? alpha : 1.0;
      mat.depthWrite = !xray;
      m.renderOrder = xray ? order : 0;
      mat.needsUpdate = true;
    }
  };

  const applyVisualMode = () => {
    const xray = selCam.value === "xray";
    setXrayMatGroup(cylHousingMeshes, xray, 0.22, 2);
    setXrayMatGroup(boilerOuterMeshes, xray, 0.14, 2);
    setXrayMatGroup(copperCoreMeshes, xray, 0.34, 1);
    setXrayMatGroup(lubeShellMeshes, xray, 0.30, 2);
    setXrayMatGroup(tenderOuterMeshes, xray, 0.22, 2);
    if (selCam.value === "cab") {
      const lb = idxMap.locoBody;
      const lx = lb >= 0 ? data.xpos[3 * lb] : 0;
      const ly = lb >= 0 ? data.xpos[3 * lb + 1] : 0;
      const lz = lb >= 0 ? data.xpos[3 * lb + 2] : 0;
      // 穿过敞口后驾驶室门洞平视全部背板阀件与 PG-ø20 / 11-7 (低于 Z=lz+0.089 车顶拱，无任何遮挡)
      camera.position.set(lx - 0.192, ly + 0.010, lz + 0.068);
      orbit.target.set(lx - 0.118, ly + 0.000, lz + 0.048);
    }
  };
  syncDomFromControls();
  applyVisualMode();

  const locoMat4 = new THREE.Matrix4();
  const tenderMat4 = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const prevLocoPos = new THREE.Vector3();
  const chaseCamLocal = new THREE.Vector3(0.28, -0.34, 0.14);
  const chaseTargetLocal = new THREE.Vector3(-0.04, 0.0, 0.055);
  const fireboxLocal = new THREE.Vector3(-0.085, 0.0, 0.038);

  selScene.addEventListener("change", () => {
    loadSimulationScene(selScene.value);
    sendWsMessage({ type: "scene", scene: isRollerMode ? "roller" : "curve" });
    pushControlsToWs();
    prevLocoPos.set(0, 0, 0);
    if (isRollerMode) {
      camera.position.set(0.28, -0.32, 0.17);
      orbit.target.set(0.02, 0.0, 0.065);
    } else {
      camera.position.set(0.26, -0.46, 0.22);
      orbit.target.set(0.0, 0.0, 0.07);
    }
    applyVisualMode();
  });
  selCam.addEventListener("change", applyVisualMode);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function syncWasmVisualJoints(dtStep: number, pGaugeKgf: number) {
    let hpTgt = 0.0;
    if (aiMode === "human" && Math.abs(humanAgent.handpumpAnimRad) > 1e-4) {
      hpTgt = Math.max(-0.36, Math.min(0.36, humanAgent.handpumpAnimRad));
    } else if (handpumpVisTimer > 0.0) {
      handpumpVisTimer = Math.max(0.0, handpumpVisTimer - dtStep);
      handpumpVisPhase += dtStep * 2.0 * Math.PI * 2.1;
      hpTgt = 0.34 * Math.sin(handpumpVisPhase);
    } else {
      handpumpVisPhase = 0.0;
    }

    const pNorm = Math.max(0.0, Math.min(1.05, pGaugeKgf / 6.0));
    const openWaterBay =
      ctrl.coalCoverOpen || ctrl.waterFillValve > 0.02 || ctrl.tenderDrainOpen || Math.abs(hpTgt) > 0.01;
    const targets: Record<string, number> = {
      j_cab_regulator: 1.12 * ctrl.regulator,
      j_cab_blower: 3.14 * ctrl.blowerValve,
      j_cab_whistle: -0.38 * ctrl.whistleValve,
      j_cab_sight_drain: ctrl.sightDrainValve ? 0.0016 : 0.0,
      j_cab_pg_needle: -2.10 + pNorm * 4.20,
      j_cab_blowdown: 3.14 * ctrl.blowdownValve,
      j_cab_fire_door: +1.65 * ctrl.fireboxDoor,
      j_cab_bypass: -1.35 * ctrl.bypassValve,
      j_cab_roof_hatch: ctrl.cabRoofOpen ? 2.10 : 0.0,
      j_tender_handpump: hpTgt,
      j_tender_hp_ram: 0.008 * hpTgt,
      j_tender_drain_plug: ctrl.tenderDrainOpen ? 0.0050 : 0.0,
      j_tender_coal_cover: openWaterBay ? 1.75 : 0.0,
      j_tender_water_lid: ctrl.waterLidOpen ? 1.45 : 0.0,
      j_tender_fuel_valve: 1.50 * ctrl.fuelValve,
      j_tender_fuel_cap: ctrl.fuelCapOpen ? 0.0045 : 0.0,
    };
    const alpha = Math.min(1.0, dtStep * 22.0);
    for (const [jn, tgt] of Object.entries(targets)) {
      const addrs = visJointMap[jn];
      if (addrs) {
        const [qadr, dadr] = addrs;
        data.qpos[qadr] = data.qpos[qadr] + alpha * (tgt - data.qpos[qadr]);
        data.qvel[dadr] = 0.0;
      }
    }
  }

  function unpackWsFrameToOutput(frame: any): StepOutput {
    const t = frame.tel;
    const sp: SpatialFluidState = frame.spatial;
    const exhL =
      Math.max(0, -(sp.valvePortOpenL[0] ?? 0)) * (sp.cylLPKgf[0] ?? 0) +
      Math.max(0, -(sp.valvePortOpenL[1] ?? 0)) * (sp.cylLPKgf[1] ?? 0);
    const exhR =
      Math.max(0, -(sp.valvePortOpenR[0] ?? 0)) * (sp.cylRPKgf[0] ?? 0) +
      Math.max(0, -(sp.valvePortOpenR[1] ?? 0)) * (sp.cylRPKgf[1] ?? 0);
    const chuffExh = Math.min(1.0, 0.32 * (exhL + exhR) + 0.45 * Math.min(1.0, t.steamGenGS / 0.35));
    const boilProg =
      typeof t.boilingProgressPct === "number"
        ? t.boilingProgressPct
        : Math.max(0.0, Math.min(100.0, ((t.boilerTempC - 25.0) / 75.0) * 100.0));

    return {
      simTime: t.simTime ?? 0.0,
      boilingProgressPct: boilProg,
      forceL: t.forceL,
      forceR: t.forceR,
      forcePump: t.fPump ?? 0.0,
      valveFrictionL: t.valveFrictionL,
      valveFrictionR: t.valveFrictionR,
      boilerPressureKgf: t.boilerPressureKgf,
      pgBourdonKgf: t.pgBourdonKgf ?? t.boilerPressureKgf,
      turretPressureKgf: t.turretPressureKgf ?? t.boilerPressureKgf,
      steamChestKgf: t.steamChestKgf,
      boilerTempC: t.boilerTempC,
      superheatTempC: t.superheatTempC,
      crownTempC: t.crownTempC,
      crownSubmergedMm: t.crownSubmergedMm,
      copperAnnealDamage: t.copperAnnealDamage,
      sealHealth: t.sealHealth,
      waterLevelPct: t.waterLevelPct,
      trueWaterLevelPct: t.trueWaterLevelPct,
      sightGlassLevelPct: t.sightGlassLevelPct ?? t.waterLevelPct,
      cylWallTempC: t.cylWallTempC,
      condensateG: t.condensateG,
      smokeboxVacuumPa: t.smokeboxVacuumPa,
      lutActive: t.lutActive,
      steamGenGS: t.steamGenGS,
      feedwaterGS: t.feedwaterGS,
      alcoholMl: t.alcoholMl,
      tenderWaterMl: t.tenderWaterMl,
      tenderWaterTempC: t.tenderWaterTempC ?? 25.0,
      returnWaterMlS: t.returnWaterMlS ?? 0.0,
      waterFillMlS: t.waterFillMlS ?? 0.0,
      tenderDrainMlS: t.tenderDrainMlS ?? 0.0,
      tenderOverflowMlS: t.tenderOverflowMlS ?? 0.0,
      mariotteHeadspaceKpa: t.mariotteHeadspaceKpa ?? 0.0,
      mariotteBubbling: !!t.mariotteBubbling,
      mariotteFloodAlarm: !!t.mariotteFloodAlarm,
      whistleSteamGS: t.whistleSteamGS ?? 0.0,
      blowdownGS: t.blowdownGS ?? 0.0,
      sightPurgeGS: t.sightPurgeGS ?? 0.0,
      lubricatorOilPct: t.lubricatorOilPct,
      wheelRpm: t.wheelRpm,
      scaleSpeedKmh: t.scaleSpeedKmh,
      safetyValveOpen: t.safetyValveOpen,
      dryFireAlarm: t.dryFireAlarm,
      flashSteamSpike: t.flashSteamSpike,
      hydrolockL: t.hydrolockL,
      hydrolockR: t.hydrolockR,
      exhaustPlume: chuffExh,
      safetyPlume: t.safetyValveOpen ? 1.0 : 0.0,
      drainPlume: ctrl.drainCock && t.steamChestKgf > 0.08 ? Math.min(1.0, 0.28 + t.condensateG * 1.4) : 0.0,
      whistlePlume: Math.min(1.0, (t.whistleSteamGS ?? 0.0) * 3.8),
      sightPurgePlume: ctrl.sightDrainValve ? Math.min(1.0, 0.35 + (t.sightPurgeGS ?? 0.0) * 0.65) : 0.0,
      blowdownPlume: ctrl.blowdownValve > 0.02 ? Math.min(1.0, 0.30 + (t.blowdownGS ?? 0.0) * 0.42) : 0.0,
      spatial: sp,
    };
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = model.opt.timestep;
    let out: StepOutput | null = null;
    const lb = idxMap.locoBody;
    const tb = idxMap.tenderBody;
    const expectedScene = isRollerMode ? "roller" : "curve";

    const useLiveGpu =
      wsConnected &&
      latestWsFrame &&
      latestWsFrame.scene === expectedScene &&
      Array.isArray(latestWsFrame.qpos) &&
      latestWsFrame.qpos.length === model.nq;

    if (useLiveGpu) {
      const qArr: number[] = latestWsFrame.qpos;
      for (let i = 0; i < qArr.length; i++) {
        data.qpos[i] = qArr[i];
      }
      mujoco.mj_forward(model, data);
      if (latestWsFrame.aiMode && latestWsFrame.aiMode !== aiMode && !activeDragPart) {
        aiMode = latestWsFrame.aiMode as AiDriverMode;
      }
      aiStatusText = latestWsFrame.aiStatus || "";
      if (aiMode !== "off" && latestWsFrame.ctrl && !activeDragPart) {
        Object.assign(ctrl, latestWsFrame.ctrl);
        syncDomFromControls();
      }
      out = unpackWsFrameToOutput(latestWsFrame);
    } else {
      for (let s = 0; s < 33; s++) {
        const sd = data.sensordata;
        const w3 = sd[idxMap.sW3];
        const vVL = data.qvel[idxMap.qvelValL];
        const vVR = data.qvel[idxMap.qvelValR];

        let pitchRad = 0.0;
        let accelX = 0.0;
        if (!isRollerMode && lb >= 0) {
          const r20 = Math.max(-0.35, Math.min(0.35, data.xmat[9 * lb + 6]));
          pitchRad = Math.asin(r20);
          const fwdVel = w3 * 0.0219;
          accelX = (fwdVel - prevFwdVel) / Math.max(1e-6, dt);
          prevFwdVel = fwdVel;
        }

        out = thermo.step(
          dt,
          sd[idxMap.sPL], sd[idxMap.sVL], sd[idxMap.sValL],
          sd[idxMap.sPR], sd[idxMap.sVR], sd[idxMap.sValR],
          w3, sd[idxMap.sPump], ctrl,
          vVL, vVR, pitchRad, accelX
        );
        data.ctrl[idxMap.actFL] = out.forceL;
        data.ctrl[idxMap.actFR] = out.forceR;
        data.ctrl[idxMap.actRev] = ctrl.reverser * 0.45;
        data.qfrc_applied[idxMap.qvelValL] = out.valveFrictionL;
        data.qfrc_applied[idxMap.qvelValR] = out.valveFrictionR;
        data.qfrc_applied[idxMap.qvelPump] = out.forcePump;
        if (idxMap.qvelDrain >= 0) {
          data.qfrc_applied[idxMap.qvelDrain] = ctrl.drainCock ? 0.0085 : 0.0;
        }
        if (isRollerMode) {
          data.qfrc_applied[idxMap.qvelW3] = -0.0022 * w3 - 0.00012 * w3 * Math.abs(w3);
        } else {
          data.qfrc_applied[idxMap.qvelW3] = -0.0055 * w3 - 0.00048 * w3 * Math.abs(w3);
          if (tb >= 0) {
            model.body_mass[tb] = tenderDryMass + (out.tenderWaterMl + out.alcoholMl * 0.789) * 1e-3;
          }
        }
        syncWasmVisualJoints(dt, out.pgBourdonKgf ?? out.boilerPressureKgf);
        mujoco.mj_step(model, data);
      }

      if (out) {
        if (aiMode === "human") {
          humanAgent.update(33 * dt, out, ctrl, (st) => triggerHandPumpBoth(st));
          aiStatusText = humanAgent.statusSummary;
          syncDomFromControls();
        } else if (aiMode === "oracle") {
          oracleAgent.update(33 * dt, out, ctrl, (st) => triggerHandPumpBoth(st));
          aiStatusText = oracleAgent.statusSummary;
          syncDomFromControls();
        } else {
          aiStatusText = "";
        }
      }
    }

    for (let g = 0; g < geomMeshes.length; g++) {
      const mesh = geomMeshes[g];
      if (!mesh.visible) continue;
      mesh.position.set(data.geom_xpos[3 * g], data.geom_xpos[3 * g + 1], data.geom_xpos[3 * g + 2]);
      const m9 = g * 9;
      locoMat4.set(
        data.geom_xmat[m9], data.geom_xmat[m9 + 1], data.geom_xmat[m9 + 2], 0,
        data.geom_xmat[m9 + 3], data.geom_xmat[m9 + 4], data.geom_xmat[m9 + 5], 0,
        data.geom_xmat[m9 + 6], data.geom_xmat[m9 + 7], data.geom_xmat[m9 + 8], 0,
        0, 0, 0, 1
      );
      mesh.quaternion.setFromRotationMatrix(locoMat4);
    }

    const lx = data.xpos[3 * lb], ly = data.xpos[3 * lb + 1], lz = data.xpos[3 * lb + 2];
    tmpQuat.set(data.xquat[4 * lb + 1], data.xquat[4 * lb + 2], data.xquat[4 * lb + 3], data.xquat[4 * lb]);
    locoMat4.compose(new THREE.Vector3(lx, ly, lz), tmpQuat, new THREE.Vector3(1, 1, 1));

    let hasTender = false;
    if (tb >= 0) {
      hasTender = true;
      const tx = data.xpos[3 * tb], ty = data.xpos[3 * tb + 1], tz = data.xpos[3 * tb + 2];
      tmpQuat.set(data.xquat[4 * tb + 1], data.xquat[4 * tb + 2], data.xquat[4 * tb + 3], data.xquat[4 * tb]);
      tenderMat4.compose(new THREE.Vector3(tx, ty, tz), tmpQuat, new THREE.Vector3(1, 1, 1));
    }

    if (selCam.value === "chase") {
      const desiredCam = chaseCamLocal.clone().applyMatrix4(locoMat4);
      const desiredTgt = chaseTargetLocal.clone().applyMatrix4(locoMat4);
      camera.position.lerp(desiredCam, 0.14);
      orbit.target.lerp(desiredTgt, 0.22);
      prevLocoPos.set(lx, ly, lz + 0.035);
    } else if (!isRollerMode) {
      const curPos = new THREE.Vector3(lx, ly, lz + 0.035);
      if (prevLocoPos.lengthSq() > 0) {
        camera.position.add(curPos.clone().sub(prevLocoPos));
      }
      orbit.target.copy(curPos);
      prevLocoPos.copy(curPos);
    } else {
      prevLocoPos.set(0, 0, 0);
    }

    if (out) {
      const sp = out.spatial;
      const xrayActive = selCam.value === "xray";
      const sd = data.sensordata;

      whistleAudio.update(ctrl.whistleValve, out.boilerPressureKgf);
      steamParticles.update(
        1 / 60,
        locoMat4,
        out.exhaustPlume,
        out.safetyPlume,
        out.drainPlume,
        sp.exhaustDryness,
        out.whistlePlume,
        out.sightPurgePlume,
        out.blowdownPlume
      );

      fluidVis.update(
        1 / 60,
        locoMat4,
        out,
        sd[idxMap.sPL],
        sd[idxMap.sPR],
        currentCalib,
        xrayActive,
        ctrl.burnerLit,
        hasTender ? tenderMat4 : null
      );

      const hotRatio = Math.max(0.0, Math.min(1.0, (out.crownTempC - 240.0) / 300.0));
      fireboxGlowLight.position.copy(fireboxLocal).applyMatrix4(locoMat4);
      fireboxGlowLight.intensity = ctrl.burnerLit ? 0.38 + 3.2 * hotRatio : 2.8 * hotRatio;
      for (const bMat of boilerShellMats) {
        bMat.emissive.setRGB(0.28 * hotRatio, 0.04 * hotRatio, 0.01 * hotRatio);
      }

      const badgeEl = document.getElementById("badge-cfd")!;
      if (sp.gpuActive) {
        const mVox = (sp.totalVoxels / 1e6).toFixed(2);
        badgeEl.textContent = sp.pino4OpActive
          ? `RTX 5080 4-PINO+6D · ${mVox}M`
          : sp.pinoActive
            ? `RTX 5080 PINO+IAPWS · ${mVox}M`
            : `RTX 5080 LIVE 3D · ${mVox}M VOX`;
        badgeEl.classList.add("live-3d");
      } else {
        badgeEl.textContent = out.lutActive ? "RTX 5080 CFD LUT" : "1D ANALYTIC";
        badgeEl.classList.remove("live-3d");
      }

      const pgRead = out.pgBourdonKgf ?? out.boilerPressureKgf;
      document.getElementById("txt-pressure")!.textContent = `${pgRead.toFixed(2)} / 4.00 kgf/cm²`;
      (document.getElementById("bar-pressure") as HTMLElement).style.width = `${Math.min(100, (pgRead / 4.5) * 100)}%`;

      const boilTxt =
        out.boilerPressureKgf > 0.02
          ? `${out.boilerTempC.toFixed(1)}°C | 蒸発中 (${out.steamGenGS.toFixed(2)}g/s)`
          : `${out.boilerTempC.toFixed(1)}°C | 沸騰進捗 ${out.boilingProgressPct.toFixed(0)}%`;
      document.getElementById("txt-boil-prog")!.textContent = boilTxt;
      (document.getElementById("bar-boil-prog") as HTMLElement).style.width = `${out.boilingProgressPct}%`;

      const sgShow = out.sightGlassLevelPct ?? out.waterLevelPct;
      const lvlDiff = Math.abs(sgShow - out.trueWaterLevelPct);
      document.getElementById("txt-water")!.textContent =
        lvlDiff > 0.6
          ? `${sgShow.toFixed(1)}% (実${out.trueWaterLevelPct.toFixed(0)}%)`
          : `${sgShow.toFixed(1)} %`;
      (document.getElementById("bar-water") as HTMLElement).style.width = `${sgShow}%`;

      document.getElementById("txt-steam")!.textContent = `${out.steamChestKgf.toFixed(2)} kgf | ${out.superheatTempC.toFixed(0)}°C`;

      const crownEl = document.getElementById("txt-crown")!;
      crownEl.textContent = `${out.crownTempC.toFixed(0)}°C | ${(out.sealHealth * 100).toFixed(0)}%`;
      crownEl.classList.toggle("val-hot", out.crownTempC >= 240.0);

      document.getElementById("txt-smokebox")!.textContent = `-${out.smokeboxVacuumPa.toFixed(0)} Pa | ${out.cylWallTempC.toFixed(0)}°C`;
      document.getElementById("txt-speed")!.textContent = `${out.wheelRpm.toFixed(0)} RPM | ${out.scaleSpeedKmh.toFixed(1)} km/h`;

      const flowTags: string[] = [];
      if ((out.waterFillMlS ?? 0) > 0.05) flowTags.push(`注+${out.waterFillMlS.toFixed(1)}`);
      if (out.returnWaterMlS > 0.02) flowTags.push(`戻${out.returnWaterMlS.toFixed(2)}`);
      if ((out.tenderDrainMlS ?? 0) > 0.05) flowTags.push(`排-${out.tenderDrainMlS.toFixed(1)}`);
      const retTag = flowTags.length > 0 ? `${flowTags.join("/")}ml/s` : `${out.tenderWaterTempC.toFixed(0)}°C`;
      document.getElementById("txt-tanks")!.textContent = `${out.alcoholMl.toFixed(0)}ml | ${out.tenderWaterMl.toFixed(0)}ml (${retTag})`;
      document.getElementById("txt-forces")!.textContent = `L: ${out.forceL.toFixed(1)}N | R: ${out.forceR.toFixed(1)}N`;

      const shExitDry = sp.shProfileDryness[sp.shProfileDryness.length - 1] ?? sp.steamDrynessDome;
      document.getElementById("txt-fluid3d")!.textContent =
        `x=${sp.steamDrynessDome.toFixed(2)}→${shExitDry.toFixed(2)} | φ=${sp.airFuelPhi.toFixed(2)} | ${sp.oilFilmUm.toFixed(1)}μm`;

      if (aiMode !== "off" && aiStatusText) {
        elAiStatus.style.display = "block";
        elAiStatus.textContent = aiStatusText;
      } else {
        elAiStatus.style.display = "none";
      }

      const copBox = document.getElementById("copilot-box") as HTMLElement;
      const cop = useLiveGpu && latestWsFrame?.copilot ? latestWsFrame.copilot : null;
      if (cop && cop.headline) {
        copBox.style.display = "block";
        const lvl = String(cop.status_level || "NOMINAL").toUpperCase();
        copBox.classList.toggle("caution", lvl === "CAUTION");
        copBox.classList.toggle("critical", lvl === "CRITICAL");
        document.getElementById("txt-copilot-model")!.textContent =
          cop.available ? `L3 ${cop.model_name} (${Math.round(cop.latency_ms ?? 0)}ms)` : "L3 Physics Copilot";
        document.getElementById("txt-copilot-lvl")!.textContent = lvl;
        document.getElementById("txt-copilot-body")!.textContent =
          `${cop.headline}${cop.recommended_action ? " — " + cop.recommended_action : ""}`;
      } else {
        copBox.style.display = "none";
      }

      let alarmMsg = "";
      if (out.hydrolockL || out.hydrolockR) {
        alarmMsg = "HYDROLOCK! 冷気筒水撃ロック発生 — 直ちに8-6ドレン弁を開放せよ!";
      } else if (out.mariotteFloodAlarm) {
        alarmMsg = "MARIOTTE FLOOD! 36-5気密蓋開放によりサンプ燃料溢流・過濃炎発生! 蓋を締めよ!";
      } else if (out.flashSteamSpike) {
        alarmMsg = "FLASH SPIKE! 高温空焚き火室への急冷給水フラッシュ超圧!";
      } else if (out.dryFireAlarm) {
        alarmMsg = `空焚き赤熱警告! 銅火室頂板 ${out.crownTempC.toFixed(0)}°C (退火損傷 ${(out.copperAnnealDamage * 100).toFixed(1)}%)`;
      } else if (out.safetyValveOpen) {
        alarmMsg = "安全弁噴出中 (4.00 kgf/cm²)";
      }
      document.getElementById("txt-alarm")!.textContent = alarmMsg;
    }

    orbit.update();
    renderer.render(scene, camera);
  }

  animate();
}

main().catch((err) => {
  loadingText.textContent = `初始化失败: ${err.message}`;
  console.error(err);
});
