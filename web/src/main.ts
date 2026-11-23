// web/src/main.ts
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
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
  cab_regulator_vis: { label: "11-11 加減弁ハンドル (上下ドラッグで蒸気開度 11-10 二耳グランド)", kind: "drag" },
  cab_reverser_wheel_vis: { label: "8-16 左運転台・ねじ式逆転機双柄クランク (ドラッグで前後進カットオフ)", kind: "drag" },
  cab_blower_vis: { label: "11-13 左上3本スポーク通風弁手輪 → 13-8 炉内ステー管 → 17-3 (ドラッグで強制ドラフト)", kind: "drag" },
  cab_whistle_vis: { label: "11-3 左上外・後方引き3穴汽笛レバー → 11-1 → 13-3 → 11-8 車底汽笛 (押下ホールドで吹鳴)", kind: "momentary" },
  cab_sight_drain_vis: { label: "13-5#3/1-1 右外縁(-Y) 11-7 水面計下部同軸ドレン栓 (クリックでフラッシュ洗浄)", kind: "toggle" },
  cab_blowdown_vis: { label: "11-16/11-17 左床下横向き(+Y)マイナス溝排水弁 (ドラッグで缶底排泥ブロー)", kind: "drag" },
  cab_fire_door_vis: { label: "12-1 低位中央・底ヒンジバタフライ火室扉 (ドラッグで開閉・二次空気流入)", kind: "drag" },
  cab_bypass_vis: { label: "4-17/6-9/8-7 左運転台(+Y)床面・黒染平滑ハブ給水バイパス弁レバー (反時計回りで 31-14 戻水量調整)", kind: "drag" },
  bypass_valve_4_17_vis: { label: "4-17 床下横置真鍮バイパスバルブ本体＋LN5-7Aロックナット＋4-15中空管ボルト (ドラッグでバイパス調整)", kind: "drag" },
  cab_drain_lever_vis: { label: "8-6 右床板上(Z=+29.4mm)・8-10双耳スタンド水平スライドローレットドレンノブ (クリックでシリンダ排液開閉)", kind: "toggle" },
  dome_lid_vis: { label: "29-12/29-13 カマボコ砂箱前部ワイヤヒンジ蓋 (クリックで開閉)", kind: "toggle" },
  drain_pull_rod_vis: { label: "8-5 右床下ドレン弁水平引き棒 (クリックで8段排液リンク開閉)", kind: "toggle" },
  tender_handpump_vis: { label: "31-3/31-5 炭水車水槽内・4節リンク手動ポンプ立柱 (クリックで圧送)", kind: "pump" },
  tender_drain_plug_vis: { label: "31-11/31-9 水槽内・上引き排水スタンドパイプ栓 (クリックで炭水車底から放水)", kind: "toggle" },
  tender_coal_cover_vis: { label: "36-1/36-2 着脱式角丸給水口蓋＋中央石炭仕切板 (クリックで着脱)", kind: "toggle" },
  tender_fuel_valve_vis: { label: "36-6 前部アルコール槽・左前T字ニードル弁 (ドラッグで 34-2 サンプ供油開度)", kind: "drag" },
  tender_fuel_cap_vis: { label: "36-5 前部アルコール槽・右気密ローレット蓋 (クリックで開閉・開弁時マリオット負圧喪失)", kind: "toggle" },
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
  const manifestResp = await fetch(`${modelBase}/manifest.json?t=${Date.now()}`, { cache: "no-cache" });
  const manifest: {
    xmls: string[];
    meshes: string[];
    configs?: string[];
    hashes?: Record<string, string>;
    build_hash?: string;
  } = await manifestResp.json();
  const assetUrl = (relPath: string): string => {
    const h = manifest.hashes?.[relPath] || manifest.build_hash || "";
    return h ? `${modelBase}/${relPath}?v=${h}` : `${modelBase}/${relPath}`;
  };
  const stlLoader = new STLLoader();
  const stlGeomMap = new Map<string, THREE.BufferGeometry>();

  let smokeboxLut: SmokeboxLutJson | null = null;
  let lvsEdgeConds: Record<string, number> = {};
  const cfgList = manifest.configs || [];
  if (cfgList.includes("smokebox_lut.json")) {
    try {
      smokeboxLut = await (await fetch(assetUrl("smokebox_lut.json"))).json();
    } catch {
      smokeboxLut = null;
    }
  }
  if (cfgList.includes("d51_fluid_lvs_report.json")) {
    try {
      const lvsJson = await (await fetch(assetUrl("d51_fluid_lvs_report.json"))).json();
      lvsEdgeConds = lvsJson?.edge_conductances || {};
    } catch {
      lvsEdgeConds = {};
    }
  }
  const humanAgent = new HumanEngineerTS();
  const oracleAgent = new IdealOracleTS();
  const whistleAudio = new BrassWhistleAudio();

  await Promise.all(
    manifest.xmls.map(async (xmlName) => {
      const txt = await (await fetch(assetUrl(xmlName))).text();
      mujoco.FS.writeFile(`/working/${xmlName}`, txt);
    })
  );

  const batchSize = 16;
  let loadedCount = 0;
  for (let i = 0; i < manifest.meshes.length; i += batchSize) {
    const chunk = manifest.meshes.slice(i, i + batchSize);
    await Promise.all(
      chunk.map(async (stlName) => {
        const buf = await (await fetch(assetUrl(`meshes/${stlName}`))).arrayBuffer();
        mujoco.FS.writeFile(`/working/meshes/${stlName}`, new Uint8Array(buf));
        const rawStl = stlLoader.parse(buf);
        rawStl.scale(0.001, 0.001, 0.001);
        const geom = toCreasedNormals(rawStl, 0.72);
        stlGeomMap.set(stlName.replace(".stl", ""), geom);
        loadedCount++;
        loadingText.textContent = `正在挂载 CAD 零件网格 (${loadedCount}/${manifest.meshes.length}): ${stlName} ...`;
      })
    );
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
  scene.add(fluidVis.worldRoot);

  let model: any = null;
  let data: any = null;
  let thermo: AsterD51Thermo1D;
  let currentCalib: CylinderCalibration;
  let geomMeshes: THREE.Mesh[] = [];
  let cylHousingMeshes: THREE.Mesh[] = [];
  let boilerOuterMeshes: THREE.Mesh[] = [];
  let insulationMeshes: THREE.Mesh[] = [];
  let copperCoreMeshes: THREE.Mesh[] = [];
  let lubeShellMeshes: THREE.Mesh[] = [];
  let tenderOuterMeshes: THREE.Mesh[] = [];
  let pipeWallXrayMeshes: THREE.Mesh[] = [];
  let boilerShellMats: THREE.MeshStandardMaterial[] = [];
  let ceramicWickMats: THREE.MeshStandardMaterial[] = [];
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
    domeLidOpen: false,
    coalCoverOpen: false,
    thermalWarp: 1.0,
    gradeDeg: 0.0,
    ambientTempC: 25.0,
    headwindMS: 0.0,
    trailingLoadKg: 0.0,
    tankSpec2010: false,
  };
  let aiMode: AiDriverMode = "off";
  let aiStatusText = "";

  let idxMap: Record<string, number> = {};
  let isRollerMode = true;
  let currentXmlFile = "d51_full.xml";
  let gpuMonitorEnabled = false;
  let locoDryMass = 3.88;
  let tenderDryMass = 1.15;
  let tenderDof0 = -1;
  let fcarSpecs: Array<{ bid: number; dof0: number; dryMass: number }> = [];
  let drvTreadGeomIds: number[] = [];
  let railGeomIds: number[] = [];
  let wasmDistanceM = 0.0;
  let wasmPrevPos: [number, number, number] | null = null;
  let prevFwdVel = 0.0;
  const wsSceneKey = () =>
    currentXmlFile === "d51_full.xml"
      ? "roller"
      : currentXmlFile === "d51_track.xml"
        ? "track"
        : currentXmlFile === "d51_oval.xml"
          ? "oval"
          : "curve";

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
  const sameOriginWs = `${wsProto}//${window.location.host}/ws`;
  const isStaticPagesHost = hostName.endsWith(".github.io");
  const wsCandidates: string[] = explicitWs
    ? [explicitWs]
    : isStaticPagesHost
      ? []
      : window.location.protocol === "https:"
        ? [sameOriginWs]
        : Array.from(new Set([sameOriginWs, `ws://${hostName}:8765/ws`, "ws://127.0.0.1:8765/ws"]));
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

  let lastWsFrameMs = 0.0;
  let reconnectTimer: number | null = null;
  const scheduleReconnect = (delayMs: number) => {
    if (reconnectTimer !== null) return;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      connectLocalGpuServer();
    }, delayMs);
  };

  const connectLocalGpuServer = () => {
    if (wsCandidates.length === 0) {
      return;
    }
    if (wsBridge && (wsBridge.readyState === WebSocket.OPEN || wsBridge.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const targetUrl = wsCandidates[wsCandidateIdx % wsCandidates.length];
    let openedOnce = false;
    try {
      const sock = new WebSocket(targetUrl);
      wsBridge = sock;
      sock.onopen = () => {
        if (wsBridge !== sock) return;
        openedOnce = true;
        wsConnected = true;
        lastWsFrameMs = performance.now();
        sendWsMessage({ type: "scene", scene: wsSceneKey() });
        if (gpuMonitorEnabled) {
          sendWsMessage({ type: "action", name: "toggleGpuMonitor", enabled: true });
        }
        pushControlsToWs();
      };
      sock.onmessage = (ev) => {
        if (wsBridge !== sock) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg && msg.type === "frame") {
            latestWsFrame = msg;
            lastWsFrameMs = performance.now();
          }
        } catch {
          // ignore malformed frame
        }
      };
      sock.onclose = () => {
        if (wsBridge === sock) {
          wsBridge = null;
          wsConnected = false;
          latestWsFrame = null;
        }
        if (!openedOnce) {
          wsCandidateIdx = (wsCandidateIdx + 1) % wsCandidates.length;
          scheduleReconnect(wsCandidateIdx === 0 ? 1500 : 120);
        } else {
          scheduleReconnect(900);
        }
      };
      sock.onerror = () => {
        sock.close();
      };
    } catch {
      wsBridge = null;
      wsConnected = false;
      scheduleReconnect(1500);
    }
  };

  // 半开死连接自愈看门狗：若前台可见状态下超过 5 秒未收到新 GPU 帧，主动重连
  window.setInterval(() => {
    if (
      wsConnected &&
      wsBridge &&
      document.visibilityState === "visible" &&
      performance.now() - lastWsFrameMs > 5000
    ) {
      wsBridge.close();
    }
  }, 2000);

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
    insulationMeshes = [];
    copperCoreMeshes = [];
    lubeShellMeshes = [];
    tenderOuterMeshes = [];
    pipeWallXrayMeshes = [];
    boilerShellMats = [];
    ceramicWickMats = [];
    copperCoreMats = [];
    interactiveParts = [];
    visJointMap = {};
    prevFwdVel = 0.0;

    model = mujoco.MjModel.loadFromXML(`/working/${xmlFile}`);
    currentXmlFile = xmlFile;
    isRollerMode = xmlFile === "d51_full.xml";
    if (isRollerMode) {
      injectRollerFlywheelInertia();
    }

    const jRev = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_reverser");
    const actRev = objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl");
    model.jnt_range[2 * jRev] = -0.36;
    model.jnt_range[2 * jRev + 1] = 0.36;
    model.actuator_ctrlrange[2 * actRev] = -0.34;
    model.actuator_ctrlrange[2 * actRev + 1] = 0.34;
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
      "j_reverser_wheel",
      "j_cab_regulator",
      "j_cab_blower",
      "j_cab_whistle",
      "j_cab_sight_drain",
      "j_cab_pg_needle",
      "j_cab_blowdown",
      "j_cab_fire_door",
      "j_cab_bypass",
      "j_cab_drain",
      "j_dome_lid",
      "j_safety_poppet_1",
      "j_safety_poppet_2",
      "j_tender_hp_link",
      "j_tender_handpump",
      "j_tender_hp_ram",
      "j_tender_drain_plug",
      "j_tender_coal_cover",
      "j_tender_fuel_valve",
      "j_tender_fuel_cap",
    ]) {
      const jid = objId(mujoco.mjtObj.mjOBJ_JOINT, jn);
      if (jid >= 0) {
        visJointMap[jn] = [model.jnt_qposadr[jid], model.jnt_dofadr[jid]];
      }
    }

    if (idxMap.locoBody >= 0) {
      locoDryMass = Math.max(1.5, model.body_mass[idxMap.locoBody] - 0.210);
    }
    if (idxMap.tenderBody >= 0) {
      tenderDryMass = Math.max(0.6, model.body_mass[idxMap.tenderBody] - 0.375);
    }
    const jTfree = objId(mujoco.mjtObj.mjOBJ_JOINT, "tender_free");
    tenderDof0 = jTfree >= 0 ? model.jnt_dofadr[jTfree] : -1;
    fcarSpecs = [];
    for (let cIdx = 1; cIdx <= 8; cIdx++) {
      const cbid = objId(mujoco.mjtObj.mjOBJ_BODY, `freight_car_${cIdx}`);
      const cjid = objId(mujoco.mjtObj.mjOBJ_JOINT, `fcar_free_${cIdx}`);
      if (cbid >= 0 && cjid >= 0) {
        fcarSpecs.push({ bid: cbid, dof0: model.jnt_dofadr[cjid], dryMass: model.body_mass[cbid] });
      }
    }
    drvTreadGeomIds = [];
    railGeomIds = [];
    for (let g = 0; g < model.ngeom; g++) {
      const gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM.value ?? mujoco.mjtObj.mjOBJ_GEOM, g) || "";
      if (gn.startsWith("rail_")) railGeomIds.push(g);
      else if (gn.startsWith("drv_") && gn.includes("_tread_")) drvTreadGeomIds.push(g);
    }
    wasmDistanceM = 0.0;
    wasmPrevPos = null;

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
      steamLap: 0.995 * leadAmp,
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
    const matCeramicWick = new THREE.MeshStandardMaterial({
      color: 0xeae3d2,
      metalness: 0.02,
      roughness: 0.94,
    });
    const matCeramicInsulation = new THREE.MeshStandardMaterial({
      color: 0xf4f1ea,
      metalness: 0.02,
      roughness: 0.95,
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
      mat_ceramic_wick: matCeramicWick,
      mat_ceramic_insulation: matCeramicInsulation,
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
              // 将 256×256 PG-ø20 白底刻度盘正交投影至严格垂直表盘面 (YZ 竖直平面，法向 [-1, 0, 0]，零仰角!)
              const uvArr = new Float32Array(posAttr.count * 2);
              for (let vi = 0; vi < posAttr.count; vi++) {
                const py = posAttr.getY(vi);
                const pz = posAttr.getZ(vi);
                const u = 0.5 - py / 0.0168;
                const v = 0.5 + pz / 0.0168;
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
        } else if (meshKey === "superheater_piping" || meshKey === "exhaust_blast_pipe") {
          mat = matSuperheater.clone();
          attachSuperheaterShader(mat as THREE.MeshStandardMaterial, fluidVis.uniforms);
        } else if (
          meshKey === "feedwater_piping" ||
          meshKey === "bypass_valve_4_17" ||
          meshKey === "boiler_riser_6_8" ||
          meshKey === "bypass_return_4_16"
        ) {
          mat = matFeedwater.clone();
          attachFeedwaterShader(mat as THREE.MeshStandardMaterial, fluidVis.uniforms, false);
        } else if (gName === "umb_hose_feed" || gName === "umb_hose_return") {
          mat = matWaterHose.clone();
          attachFeedwaterShader(mat as THREE.MeshStandardMaterial, fluidVis.uniforms, true);
        } else if (gName === "umb_hose_fuel") {
          mat = matSiliconeHose.clone();
        } else if (INTERACTIVE_PART_META[gName]) {
          mat = baseMat.clone();
        } else if (meshKey === "displacement_lubricator") {
          mat = baseMat.clone();
        } else if (meshKey === "boiler_superstructure") {
          const bMat = baseMat.clone();
          boilerShellMats.push(bMat);
          mat = bMat;
        } else if (meshKey === "boiler_insulation") {
          mat = baseMat.clone();
        } else if (meshKey === "burner_wick") {
          const wMat = baseMat.clone();
          ceramicWickMats.push(wMat);
          mat = wMat;
        } else if (
          meshKey === "cylinder_block_L" ||
          meshKey === "cylinder_block_R" ||
          meshKey === "tender_body" ||
          meshKey === "tender_internals" ||
          meshKey === "cab_backhead_plate" ||
          meshKey === "firebox_wrapper_12_4" ||
          meshKey === "boiler_aux_valves" ||
          meshKey === "alcohol_burner" ||
          meshKey === "blower_nozzle_17_3"
        ) {
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
        } else if (meshKey === "boiler_insulation") {
          insulationMeshes.push(mesh);
        } else if (meshKey === "boiler_firebox_flues") {
          copperCoreMeshes.push(mesh);
        } else if (meshKey === "displacement_lubricator") {
          lubeShellMeshes.push(mesh);
        } else if (meshKey === "tender_body" || meshKey === "tender_coal_cover") {
          tenderOuterMeshes.push(mesh);
        } else if (
          meshKey === "tender_internals" ||
          meshKey === "cab_backhead_plate" ||
          meshKey === "firebox_wrapper_12_4" ||
          meshKey === "boiler_aux_valves" ||
          meshKey === "alcohol_burner" ||
          meshKey === "blower_nozzle_17_3" ||
          meshKey === "superheater_piping" ||
          meshKey === "exhaust_blast_pipe" ||
          meshKey === "feedwater_piping" ||
          meshKey === "bypass_valve_4_17" ||
          meshKey === "boiler_riser_6_8" ||
          meshKey === "bypass_return_4_16" ||
          gName === "umb_hose_fuel" ||
          gName === "umb_hose_feed" ||
          gName === "umb_hose_return"
        ) {
          pipeWallXrayMeshes.push(mesh);
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
  const elLoad = document.getElementById("rng-load") as HTMLInputElement | null;
  const elAmb = document.getElementById("rng-amb") as HTMLInputElement | null;
  const selAiMode = document.getElementById("sel-ai-mode") as HTMLSelectElement;
  const elAiStatus = document.getElementById("txt-ai-status")!;
  const btnBurner = document.getElementById("btn-burner")!;
  const btnFan = document.getElementById("btn-fan")!;
  const btnDrain = document.getElementById("btn-drain")!;
  const btnWhistle = document.getElementById("btn-whistle")!;
  const btnSightDrain = document.getElementById("btn-sightdrain")!;
  const btnFireDoor = document.getElementById("btn-firedoor")!;
  const btnBlowdown = document.getElementById("btn-blowdown")!;
  const btnDomeLid = document.getElementById("btn-domelid")!;
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
    if (elLoad) {
      const ld = ctrl.trailingLoadKg ?? 0.0;
      elLoad.value = ld.toFixed(1);
      document.getElementById("val-load")!.textContent = `${ld.toFixed(1)}kg`;
    }
    if (elAmb) {
      const amb = ctrl.ambientTempC ?? 25.0;
      elAmb.value = amb.toFixed(0);
      document.getElementById("val-amb")!.textContent = `${amb.toFixed(0)}°C`;
    }

    btnBurner.classList.toggle("active", ctrl.burnerLit);
    btnFan.classList.toggle("active", ctrl.chimneyFan);
    btnDrain.classList.toggle("active", ctrl.drainCock);
    btnWhistle.classList.toggle("active", ctrl.whistleValve > 0.1);
    btnSightDrain.classList.toggle("active", ctrl.sightDrainValve);
    btnFireDoor.classList.toggle("active", ctrl.fireboxDoor > 0.1);
    btnBlowdown.classList.toggle("active", ctrl.blowdownValve > 0.1);
    btnDomeLid.classList.toggle("active", !!ctrl.domeLidOpen);
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
  if (elLoad) {
    bindRange("rng-load", "val-load", (v) => `${v.toFixed(1)}kg`, (v) => (ctrl.trailingLoadKg = v), false);
  }
  if (elAmb) {
    bindRange("rng-amb", "val-amb", (v) => `${v.toFixed(0)}°C`, (v) => (ctrl.ambientTempC = v), false);
  }

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
  btnDomeLid.addEventListener("click", () => {
    ctrl.domeLidOpen = !ctrl.domeLidOpen;
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
      domeLidOpen: false,
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
      domeLidOpen: false,
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
      case "cab_bypass_vis":
      case "bypass_valve_4_17_vis": return `${Math.round(ctrl.bypassValve * 100)}%`;
      case "cab_drain_lever_vis": return ctrl.drainCock ? "開 (排液)" : "閉";
      case "dome_lid_vis": return ctrl.domeLidOpen ? "開" : "閉";
      case "drain_pull_rod_vis": return ctrl.drainCock ? "開 (排液)" : "閉";
      case "tender_handpump_vis": return "クリック圧送";
      case "tender_drain_plug_vis": return ctrl.tenderDrainOpen ? "放水中" : "閉";
      case "tender_coal_cover_vis": return ctrl.coalCoverOpen ? "開" : "閉";
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
      else if (gn === "cab_bypass_vis" || gn === "bypass_valve_4_17_vis") ctrl.bypassValve = Math.max(0, Math.min(1, dragStartVal + delta));
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
        (gn === "cab_bypass_vis" || gn === "bypass_valve_4_17_vis") ? ctrl.bypassValve :
        gn === "cab_blowdown_vis" ? ctrl.blowdownValve :
        gn === "tender_fuel_valve_vis" ? ctrl.fuelValve :
        ctrl.fireboxDoor;
    } else if (hit.kind === "momentary") {
      activeDragPart = hit;
      setWhistleState(true);
    } else if (hit.kind === "pump") {
      triggerHandPumpBoth(2);
    } else if (hit.kind === "toggle") {
      if (gn === "drain_pull_rod_vis" || gn === "cab_drain_lever_vis") {
        if (aiMode !== "off") setAiModeLocal("off", false);
        ctrl.drainCock = !ctrl.drainCock;
      } else if (gn === "cab_sight_drain_vis") {
        ctrl.sightDrainValve = !ctrl.sightDrainValve;
      } else if (gn === "dome_lid_vis") {
        ctrl.domeLidOpen = !ctrl.domeLidOpen;
      } else if (gn === "tender_drain_plug_vis") {
        ctrl.tenderDrainOpen = !ctrl.tenderDrainOpen;
      } else if (gn === "tender_coal_cover_vis") {
        ctrl.coalCoverOpen = !ctrl.coalCoverOpen;
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
  const btnFlowDir = document.getElementById("btn-flow-dir");
  const btnPipeLabels = document.getElementById("btn-pipe-labels");
  const pipeLabelsLayer = document.getElementById("pipe-labels-layer")!;
  let flowDirEnabled = false;
  let pipeLabelsEnabled = false;

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

  const syncOverlayButtons = () => {
    btnFlowDir?.classList.toggle("active", flowDirEnabled);
    btnPipeLabels?.classList.toggle("active", pipeLabelsEnabled);
    const thermalBar = document.getElementById("thermal-scale-bar");
    if (thermalBar) {
      thermalBar.style.display = (selCam.value === "xray" || flowDirEnabled) ? "flex" : "none";
    }
    if (!pipeLabelsEnabled) {
      pipeLabelsLayer.innerHTML = "";
    }
  };

  const applyVisualMode = () => {
    const xray = selCam.value === "xray";
    setXrayMatGroup(cylHousingMeshes, xray, 0.20, 2);
    setXrayMatGroup(boilerOuterMeshes, xray, 0.12, 2);
    setXrayMatGroup(insulationMeshes, xray, 0.20, 2);
    setXrayMatGroup(copperCoreMeshes, xray, 0.30, 1);
    setXrayMatGroup(lubeShellMeshes, xray, 0.28, 2);
    setXrayMatGroup(tenderOuterMeshes, xray, 0.18, 2);
    setXrayMatGroup(pipeWallXrayMeshes, xray, 0.30, 3);
    if (xray && !pipeLabelsEnabled) {
      pipeLabelsEnabled = true;
    } else if (!xray && !flowDirEnabled) {
      pipeLabelsEnabled = false;
    }
    syncOverlayButtons();
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
  const whistleMouthLocal = new THREE.Vector3(-0.102, 0.0202, 0.0232);
  const prevWhistleWorld = new THREE.Vector3();
  let hasPrevWhistlePos = false;

  const btnGpuMon = document.getElementById("btn-gpu-mon");
  if (btnGpuMon) {
    btnGpuMon.addEventListener("click", () => {
      gpuMonitorEnabled = !gpuMonitorEnabled;
      btnGpuMon.classList.toggle("active", gpuMonitorEnabled);
      sendWsMessage({ type: "action", name: "toggleGpuMonitor", enabled: gpuMonitorEnabled });
    });
  }
  btnFlowDir?.addEventListener("click", () => {
    flowDirEnabled = !flowDirEnabled;
    if (flowDirEnabled && selCam.value === "normal") {
      selCam.value = "xray";
      applyVisualMode();
    } else {
      syncOverlayButtons();
    }
  });
  btnPipeLabels?.addEventListener("click", () => {
    pipeLabelsEnabled = !pipeLabelsEnabled;
    if (pipeLabelsEnabled && selCam.value === "normal") {
      selCam.value = "xray";
      applyVisualMode();
    } else {
      syncOverlayButtons();
    }
  });

  selScene.addEventListener("change", () => {
    loadSimulationScene(selScene.value);
    sendWsMessage({ type: "scene", scene: wsSceneKey() });
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

  let hpPsiSmooth = 0.0;
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

    const alpha = Math.min(1.0, dtStep * 22.0);
    hpPsiSmooth = hpPsiSmooth + alpha * (hpTgt - hpPsiSmooth);
    const sinQL = Math.max(-0.99, Math.min(0.99, 0.5 * (1.0 - Math.cos(hpPsiSmooth))));
    const qLink = Math.asin(sinQL);
    const qLev = hpPsiSmooth - qLink;
    const qRam = 0.020 * (Math.cos(qLink) - 1.0) - 0.010 * Math.sin(hpPsiSmooth);
    for (const [jnHp, valHp] of [
      ["j_tender_hp_link", qLink],
      ["j_tender_handpump", qLev],
      ["j_tender_hp_ram", qRam],
    ] as const) {
      const addrsHp = visJointMap[jnHp];
      if (addrsHp) {
        data.qpos[addrsHp[0]] = valHp;
        data.qvel[addrsHp[1]] = 0.0;
      }
    }

    const pNorm = Math.max(0.0, Math.min(1.05, pGaugeKgf / 6.0));
    const openWaterBay =
      ctrl.coalCoverOpen || ctrl.waterFillValve > 0.02 || ctrl.tenderDrainOpen || Math.abs(hpTgt) > 0.01;
    const svLiftWasm = pGaugeKgf >= 3.98 ? 0.00056 : 0.0;
    const targets: Record<string, number> = {
      j_reverser_wheel: 12.56 * ctrl.reverser,
      j_cab_regulator: 1.15 * ctrl.regulator,
      j_cab_blower: 3.14 * ctrl.blowerValve,
      j_cab_whistle: +0.38 * ctrl.whistleValve,
      j_cab_sight_drain: ctrl.sightDrainValve ? 0.0016 : 0.0,
      j_cab_pg_needle: -2.10 + pNorm * 4.20,
      j_cab_blowdown: 3.14 * ctrl.blowdownValve,
      j_cab_fire_door: +1.40 * ctrl.fireboxDoor,
      j_cab_bypass: +1.30 * ctrl.bypassValve,
      j_cab_drain: ctrl.drainCock ? -0.0038 : 0.0038,
      j_dome_lid: ctrl.domeLidOpen ? 1.15 : 0.0,
      j_safety_poppet_1: svLiftWasm,
      j_safety_poppet_2: svLiftWasm * 0.94,
      j_tender_drain_plug: ctrl.tenderDrainOpen ? 0.0050 : 0.0,
      j_tender_coal_cover: openWaterBay ? 0.026 : 0.0,
      j_tender_fuel_valve: 1.50 * ctrl.fuelValve,
      j_tender_fuel_cap: ctrl.fuelCapOpen ? 0.0045 : 0.0,
    };
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
    if (typeof t.waterFreeZMm === "number") {
      sp.waterFreeZMm = t.waterFreeZMm;
    }
    const portExhMag =
      Math.max(0, -(sp.valvePortOpenL[0] ?? 0)) +
      Math.max(0, -(sp.valvePortOpenL[1] ?? 0)) +
      Math.max(0, -(sp.valvePortOpenR[0] ?? 0)) +
      Math.max(0, -(sp.valvePortOpenR[1] ?? 0));
    const exhMassGS = t.exhaustGS ?? 0.0;
    const blwMassGS = t.blowerGS ?? 0.0;
    const chuffExh = Math.min(
      1.0,
      1.65 * exhMassGS +
        2.40 * blwMassGS +
        (t.steamChestKgf > 0.04 ? (0.16 + 0.36 * portExhMag) * Math.min(1.0, t.steamChestKgf / 1.4) : 0.0)
    );
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
      sightGlassMeniscusZMm: t.sightGlassMeniscusZMm,
      sightGlassVelMmS: t.sightGlassVelMmS ?? 0.0,
      sightGlassLumenConnected: t.sightGlassLumenConnected ?? sp.sightGlassLumenConnected ?? true,
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
      exhaustGS: exhMassGS,
      blowerGS: blwMassGS,
      safetyGS: t.safetyGS ?? 0.0,
      boilerWaterMl: t.boilerWaterMl,
      wickCapillaryMlS: t.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0,
      wickSaturation: t.wickSaturation ?? sp.wickSaturation ?? 0.90,
      wickTipTempC: t.wickTipTempC ?? sp.wickTipTempC ?? 25.0,
      casingTempC: t.casingTempC ?? sp.casingTempC ?? 25.0,
      shellHeatLossW: t.shellHeatLossW ?? sp.shellHeatLossW ?? 0.0,
      shellRadLossW: t.shellRadLossW ?? sp.shellRadLossW ?? 0.0,
      boilerEfficiencyPct: t.boilerEfficiencyPct ?? sp.boilerEfficiencyPct ?? 0.0,
      burnerLumenConnected: t.burnerLumenConnected ?? sp.burnerLumenConnected ?? true,
      steamOutGS: t.steamOutGS ?? t.steamGenGS,
      cutoffPct: t.cutoffPct ?? 0.0,
      cylExpansionWorkW: t.cylExpansionWorkW ?? 0.0,
      specificSteamGKj: t.specificSteamGKj ?? 0.0,
      pumpChamberKgf: t.pumpChamberKgf ?? sp.pumpChamberPKgf ?? 0.0,
      pumpSuctionVacKpa: t.pumpSuctionVacKpa ?? 0.0,
      pumpShaftPowerW: t.pumpShaftPowerW ?? 0.0,
      lubeOilTempC: t.lubeOilTempC ?? 25.0,
      lubeOilViscosityMpas: t.lubeOilViscosityMpas ?? 1850.0,
      alcoholTempC: t.alcoholTempC ?? 25.0,
      alcoholViscosityMpas: t.alcoholViscosityMpas ?? 1.08,
      drainBlowdownGS: t.drainBlowdownGS ?? 0.0,
      flueDeflectionMm: t.flueDeflectionMm ?? sp.flueDeflectionMm ?? [0, 0, 0, 0, 0],
      flueOvalizationMm: t.flueOvalizationMm ?? sp.flueOvalizationMm ?? [0, 0, 0, 0, 0],
      crownBulgeMm: t.crownBulgeMm ?? sp.crownBulgeMm ?? 0.0,
      maxBoilerStressMpa: t.maxBoilerStressMpa ?? sp.maxBoilerStressMpa ?? 0.0,
      copperYieldMpa: t.copperYieldMpa ?? sp.copperYieldMpa ?? 185.0,
      springDeflectMm: t.springDeflectMm ?? [],
      boilerExpansionMm: t.boilerExpansionMm,
      lubeSettlingMmS: t.lubeSettlingMmS,
      qFuelW: t.qFuelW,
      qEvapW: t.qEvapW,
      qShW: t.qShW,
      qSensibleW: t.qSensibleW,
      qFlueLossW: t.qFlueLossW,
      qCasingLossW: t.qCasingLossW,
      qUnburnedW: t.qUnburnedW,
      energyBalanceErrPct: t.energyBalanceErrPct,
      boilerType: t.boilerType ?? "Type-C",
      fireboxLinerTempC: t.fireboxLinerTempC,
      slideValveLiftMm: t.slideValveLiftMm,
      adhesionFactor: t.adhesionFactor,
      whistleFreqHz: t.whistleFreqHz,
      whistleSplDb: t.whistleSplDb,
      whistleDrynessX: t.whistleDrynessX,
      whistleWallTempC: t.whistleWallTempC,
      whistleSputter: t.whistleSputter,
      whistleJetMach: t.whistleJetMach,
      whistleSoundSpeedMS: t.whistleSoundSpeedMS,
      fireboxVacuumPa: t.fireboxVacuumPa ?? sp.fireboxVacuumPa ?? 0.0,
      grateAirGS: t.grateAirGS ?? sp.grateAirGS,
      fireholeAirGS: t.fireholeAirGS ?? sp.fireholeAirGS,
      flueGasGS: t.flueGasGS ?? sp.flueGasGS,
      smokeboxGasTempC: t.smokeboxGasTempC ?? sp.smokeboxGasTempC,
      stackExitTempC: t.stackExitTempC ?? sp.stackExitTempC,
      flameTiltDeg: t.flameTiltDeg ?? sp.flameTiltDeg,
      wickTiltDeg3: t.wickTiltDeg3 ?? sp.wickTiltDeg3,
      backdraftAlarm: !!(t.backdraftAlarm ?? sp.backdraftAlarm),
      derailed: !!t.derailed,
      derailReason: t.derailReason ?? "",
      pino7OpActive: !!(t.pino7OpActive ?? sp.pino7OpActive),
      gpuMonitor: t.gpuMonitor ?? sp.gpuMonitor ?? {},
      lubricatorOilPct: t.lubricatorOilPct,
      wheelRpm: t.wheelRpm,
      scaleSpeedKmh: t.scaleSpeedKmh,
      groundSpeedMS: t.groundSpeedMS ?? 0.0,
      groundSpeedKmh: t.groundSpeedKmh ?? t.scaleSpeedKmh,
      wheelSlipPct: t.wheelSlipPct ?? 0.0,
      railAdhesionMu: t.railAdhesionMu ?? 0.26,
      exhaustBackpressureKgf: t.exhaustBackpressureKgf ?? 0.0,
      trainResistanceN: t.trainResistanceN ?? 0.0,
      distanceM: t.distanceM ?? 0.0,
      scaleDistanceKm: t.scaleDistanceKm ?? 0.0,
      lapCount: t.lapCount ?? 0.0,
      predictedRangeM: t.predictedRangeM ?? 0.0,
      predictedRangeP10M: t.predictedRangeP10M ?? 0.0,
      predictedRangeP90M: t.predictedRangeP90M ?? 0.0,
      predictedTimeMin: t.predictedTimeMin ?? 0.0,
      rangeLimitingFactor: t.rangeLimitingFactor ?? "NOMINAL",
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
    const expectedScene = wsSceneKey();

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
        let vFwdGnd: number | undefined = undefined;
        if (!isRollerMode && lb >= 0) {
          const eFx = data.xmat[9 * lb + 0];
          const eFy = data.xmat[9 * lb + 3];
          const r20 = Math.max(-0.35, Math.min(0.35, data.xmat[9 * lb + 6]));
          pitchRad = Math.asin(r20);
          vFwdGnd = data.qvel[0] * eFx + data.qvel[1] * eFy + data.qvel[2] * r20;
          accelX = (vFwdGnd - prevFwdVel) / Math.max(1e-6, dt);
          prevFwdVel = vFwdGnd;
        }

        out = thermo.step(
          dt,
          sd[idxMap.sPL], sd[idxMap.sVL], sd[idxMap.sValL],
          sd[idxMap.sPR], sd[idxMap.sVR], sd[idxMap.sValR],
          w3, sd[idxMap.sPump], ctrl,
          vVL, vVR, pitchRad, accelX, vFwdGnd
        );
        data.ctrl[idxMap.actFL] = out.forceL;
        data.ctrl[idxMap.actFR] = out.forceR;
        data.ctrl[idxMap.actRev] = ctrl.reverser * 0.34;
        data.qfrc_applied[idxMap.qvelValL] = out.valveFrictionL;
        data.qfrc_applied[idxMap.qvelValR] = out.valveFrictionR;
        data.qfrc_applied[idxMap.qvelPump] = out.forcePump;
        if (idxMap.qvelDrain >= 0) {
          data.qfrc_applied[idxMap.qvelDrain] = ctrl.drainCock ? 0.022 : 0.0;
        }
        if (isRollerMode) {
          data.qfrc_applied[idxMap.qvelW3] = -0.0022 * w3 - 0.00012 * w3 * Math.abs(w3);
        } else {
          const mLocoCur = locoDryMass + (out.boilerWaterMl ?? 210.0) * 1e-3;
          model.body_mass[lb] = mLocoCur;
          const extraLoadKg = Math.max(0.0, ctrl.trailingLoadKg ?? 0.0);
          let mTdrCur = 0.0;
          if (tb >= 0) {
            mTdrCur = tenderDryMass + (out.tenderWaterMl * 0.997 + out.alcoholMl * 0.787) * 1e-3 + (fcarSpecs.length === 0 ? extraLoadKg : 0.0);
            model.body_mass[tb] = mTdrCur;
          }
          let mCarsTot = 0.0;
          if (fcarSpecs.length > 0) {
            const perCar = extraLoadKg / fcarSpecs.length;
            for (const fc of fcarSpecs) {
              const mc = fc.dryMass + perCar;
              model.body_mass[fc.bid] = mc;
              mCarsTot += mc;
            }
          }
          const totalConsistKg = mLocoCur + mTdrCur + mCarsTot + 2.15;

          const vx = data.qvel[0];
          const vy = data.qvel[1];
          const vSpd = Math.hypot(vx, vy);
          const vSlip = w3 * 0.0219 - (vFwdGnd ?? 0.0);
          const wetPen = (ctrl.drainCock ? 0.50 : 0.0) + Math.min(0.08, 0.03 * (out.exhaustGS ?? 0.0));
          const muRail = Math.max(0.11, Math.min(0.33, 0.33 * (1.0 - Math.min(0.54, wetPen)) * (0.72 + 0.28 * Math.exp(-Math.abs(vSlip) / 0.32))));
          for (const gid of drvTreadGeomIds) model.geom_friction[3 * gid] = muRail;
          for (const gid of railGeomIds) model.geom_friction[3 * gid] = muRail;

          const px = data.xpos[3 * lb], py = data.xpos[3 * lb + 1], pz = data.xpos[3 * lb + 2];
          // 第一性原理整车运行阻力 + 弯道轮对弦差阻力 + 勾配加算切向重力 (严禁硬编码 8.80*v^2!)
          const tAmb = ctrl.ambientTempC ?? 25.0;
          const brgVisc = Math.exp(920.0 * (1.0 / Math.max(255.0, tAmb + 273.15) - 1.0 / 298.15));
          const fRoll = (0.0028 + 0.00009 * fcarSpecs.length) * (totalConsistKg * 9.81) * Math.tanh(vSpd / 0.006);
          const fJournal = 0.048 * (totalConsistKg / 6.0) * brgVisc * vSpd;
          const rhoAir = 101325.0 / (287.05 * Math.max(250.0, tAmb + 273.15));
          const fAero = 0.5 * rhoAir * (0.82 + 0.16 * fcarSpecs.length) * 0.0112 * vSpd * vSpd;
          const curvAbs = currentXmlFile === "d51_r2m.xml" ? 0.50 : (currentXmlFile === "d51_oval.xml" && Math.abs(px) > 3.0 ? 0.40 : 0.0);
          const fCurve = (0.0096 * curvAbs) * (totalConsistKg * 9.81) * Math.tanh(vSpd / 0.008);
          const fRes = fRoll + fJournal + fAero + fCurve;

          const sinGrade = Math.sin((Math.max(-15.0, Math.min(15.0, ctrl.gradeDeg)) * Math.PI) / 180.0);
          const ex0 = data.xmat[9 * lb + 0], ey0 = data.xmat[9 * lb + 3];
          const en0 = Math.max(1e-6, Math.hypot(ex0, ey0));
          let fxL = -(mLocoCur + 1.55) * 9.81 * sinGrade * (ex0 / en0);
          let fyL = -(mLocoCur + 1.55) * 9.81 * sinGrade * (ey0 / en0);
          if (vSpd > 1e-5) {
            fxL -= fRes * (vx / vSpd);
            fyL -= fRes * (vy / vSpd);
          }
          data.qfrc_applied[0] = fxL;
          data.qfrc_applied[1] = fyL;
          if (tenderDof0 >= 0 && tb >= 0) {
            const tx0 = data.xmat[9 * tb + 0], ty0 = data.xmat[9 * tb + 3];
            const tn0 = Math.max(1e-6, Math.hypot(tx0, ty0));
            data.qfrc_applied[tenderDof0] = -(mTdrCur + 0.34) * 9.81 * sinGrade * (tx0 / tn0);
            data.qfrc_applied[tenderDof0 + 1] = -(mTdrCur + 0.34) * 9.81 * sinGrade * (ty0 / tn0);
          }
          for (const fc of fcarSpecs) {
            const cx0 = data.xmat[9 * fc.bid + 0], cy0 = data.xmat[9 * fc.bid + 3];
            const cn0 = Math.max(1e-6, Math.hypot(cx0, cy0));
            const mcTot = model.body_mass[fc.bid] + 0.13;
            data.qfrc_applied[fc.dof0] = -mcTot * 9.81 * sinGrade * (cx0 / cn0);
            data.qfrc_applied[fc.dof0 + 1] = -mcTot * 9.81 * sinGrade * (cy0 / cn0);
          }
          data.qfrc_applied[idxMap.qvelW3] = -0.0031 * w3 - 0.00018 * w3 * Math.abs(w3);
          if (wasmPrevPos) {
            const ds = Math.hypot(px - wasmPrevPos[0], py - wasmPrevPos[1], pz - wasmPrevPos[2]);
            if (ds < 0.25) wasmDistanceM += ds;
          }
          wasmPrevPos = [px, py, pz];
          const lapLen = currentXmlFile === "d51_oval.xml" ? 27.708 : currentXmlFile === "d51_r2m.xml" ? 12.566 : 0.0;
          out.distanceM = wasmDistanceM;
          out.scaleDistanceKm = (wasmDistanceM * 30.0) / 1000.0;
          out.lapCount = lapLen > 0 ? wasmDistanceM / lapLen : 0.0;
          const remFuelS = out.alcoholMl / Math.max(0.035, out.wickCapillaryMlS || 0.105);
          const remWaterS = (out.tenderWaterMl + Math.max(0.0, (out.boilerWaterMl ?? 210.0) - 95.0)) / Math.max(0.08, out.steamGenGS || 0.24);
          const remS = Math.min(remFuelS, remWaterS);
          const vEst = Math.max(0.45, Math.abs(vFwdGnd ?? 0.65));
          out.predictedRangeM = remS * vEst * 0.88;
          out.predictedTimeMin = remS / 60.0;
          out.rangeLimitingFactor = remFuelS <= remWaterS ? "FUEL_DEPLETED" : "WATER_DEPLETED";
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
      if (!sp.edgeConductances || Object.keys(sp.edgeConductances).length === 0) {
        sp.edgeConductances = lvsEdgeConds;
      }
      const xrayActive = selCam.value === "xray";
      const sd = data.sensordata;

      const whiWorld = whistleMouthLocal.clone().applyMatrix4(locoMat4);
      const toCam = camera.position.clone().sub(whiWorld);
      const distWhM = Math.max(0.18, toCam.length());
      let dopplerFactor = 1.0;
      if (hasPrevWhistlePos) {
        const vWhWorld = whiWorld.clone().sub(prevWhistleWorld).multiplyScalar(60.0);
        const vRadial = vWhWorld.dot(toCam.normalize());
        dopplerFactor = 343.0 / Math.max(295.0, Math.min(395.0, 343.0 - vRadial));
      }
      prevWhistleWorld.copy(whiWorld);
      hasPrevWhistlePos = true;

      whistleAudio.update({
        whistleValve: ctrl.whistleValve,
        boilerPressureKgf: out.boilerPressureKgf,
        freqHz: out.whistleFreqHz,
        splDb: out.whistleSplDb,
        drynessX: out.whistleDrynessX,
        wallTempC: out.whistleWallTempC,
        sputter: out.whistleSputter,
        jetMach: out.whistleJetMach,
        soundSpeedMS: out.whistleSoundSpeedMS,
        distanceM: distWhM,
        dopplerFactor,
      });
      const fluePlume =
        ctrl.burnerLit && (out.flueGasGS ?? sp.flueGasGS ?? 0) > 0.25
          ? Math.min(1.0, ((out.flueGasGS ?? sp.flueGasGS ?? 0) - 0.20) * 0.85)
          : 0.0;
      steamParticles.update(
        1 / 60,
        locoMat4,
        out.exhaustPlume,
        out.safetyPlume,
        out.drainPlume,
        sp.exhaustDryness,
        out.whistlePlume,
        out.sightPurgePlume,
        out.blowdownPlume,
        fluePlume,
        !!(out.backdraftAlarm ?? sp.backdraftAlarm)
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
        hasTender ? tenderMat4 : null,
        flowDirEnabled,
        pipeLabelsEnabled
      );

      const hotRatio = Math.max(0.0, Math.min(1.0, (out.crownTempC - 240.0) / 300.0));
      fireboxGlowLight.position.copy(fireboxLocal).applyMatrix4(locoMat4);
      fireboxGlowLight.intensity = ctrl.burnerLit ? 0.38 + 3.2 * hotRatio : 2.8 * hotRatio;
      for (const bMat of boilerShellMats) {
        bMat.emissive.setRGB(0.28 * hotRatio, 0.04 * hotRatio, 0.01 * hotRatio);
      }
      // セラミックヤーン多孔陶瓷灯芯湿润浸醇色泽与断醇干烧 (>380°C) 赤热辐射光
      const wickSat = out.wickSaturation ?? sp.wickSaturation ?? 0.85;
      const wickTipC = out.wickTipTempC ?? sp.wickTipTempC ?? 82.0;
      const wickDryGlow = Math.max(0.0, Math.min(1.0, (wickTipC - 380.0) / 260.0));
      for (const wMat of ceramicWickMats) {
        const wetTint = 0.92 - 0.14 * wickSat;
        wMat.color.setRGB(wetTint, wetTint * 0.97, wetTint * 0.90);
        wMat.emissive.setRGB(0.95 * wickDryGlow, 0.24 * wickDryGlow, 0.03 * wickDryGlow);
      }

      const badgeEl = document.getElementById("badge-cfd")!;
      if (sp.gpuActive) {
        const mVox = (sp.totalVoxels / 1e6).toFixed(2);
        badgeEl.textContent = sp.pino7OpActive || out.pino7OpActive
          ? `RTX 5080 7-PINO+8D · ${mVox}M`
          : sp.pino6OpActive
            ? `RTX 5080 6-PINO+7D · ${mVox}M`
            : sp.pino4OpActive
              ? `RTX 5080 4-PINO+6D · ${mVox}M`
              : sp.pinoActive
                ? `RTX 5080 PINO+IAPWS · ${mVox}M`
                : `RTX 5080 LIVE 3D · ${mVox}M VOX`;
        badgeEl.title = "已连接本地 RTX 5080 实时 5.78M 体素 + 七算子 PINO + 热弹塑性形变与轮轨接触物理推流服务";
        badgeEl.classList.add("live-3d");
      } else {
        badgeEl.textContent = out.lutActive ? "OFFLINE · CFD LUT" : "1D ANALYTIC";
        badgeEl.title = "未连接本地 gpu_3d_server WebSocket，当前运行浏览器离线 WASM + CFD LUT 兜底求解器";
        badgeEl.classList.remove("live-3d");
      }

      const pgRead = out.pgBourdonKgf ?? out.boilerPressureKgf;
      document.getElementById("txt-pressure")!.textContent = `${pgRead.toFixed(2)} / 4.00 kgf/cm²`;
      (document.getElementById("bar-pressure") as HTMLElement).style.width = `${Math.min(100, (pgRead / 4.5) * 100)}%`;

      const bWaterMl = out.boilerWaterMl;
      const isDryZeroMl = bWaterMl !== undefined && bWaterMl < 0.25;
      const boilTxt = isDryZeroMl
        ? `${out.boilerTempC.toFixed(1)}°C | 完全蒸発・空焚き (0.00g/s)`
        : out.boilerPressureKgf > 0.02
          ? `${out.boilerTempC.toFixed(1)}°C | 蒸発中 (${out.steamGenGS.toFixed(2)}g/s)`
          : `${out.boilerTempC.toFixed(1)}°C | 沸騰進捗 ${out.boilingProgressPct.toFixed(0)}%`;
      document.getElementById("txt-boil-prog")!.textContent = boilTxt;
      (document.getElementById("bar-boil-prog") as HTMLElement).style.width = `${out.boilingProgressPct}%`;

      const sgShow = out.sightGlassLevelPct ?? out.waterLevelPct;
      const lvlDiff = Math.abs(sgShow - out.trueWaterLevelPct);
      document.getElementById("txt-water")!.textContent =
        bWaterMl !== undefined && bWaterMl < 139.5
          ? bWaterMl < 0.5
            ? `0.0% (完全蒸発 0ml)`
            : `0.0% (空焚き残 ${bWaterMl.toFixed(0)}ml)`
          : lvlDiff > 0.6
            ? `${sgShow.toFixed(1)}% (実${out.trueWaterLevelPct.toFixed(0)}%)`
            : `${sgShow.toFixed(1)} %`;
      (document.getElementById("bar-water") as HTMLElement).style.width = `${sgShow}%`;

      document.getElementById("txt-steam")!.textContent = `${out.steamChestKgf.toFixed(2)} kgf | ${out.superheatTempC.toFixed(0)}°C`;

      const crownEl = document.getElementById("txt-crown")!;
      crownEl.textContent = `${out.crownTempC.toFixed(0)}°C | ${(out.sealHealth * 100).toFixed(0)}%`;
      crownEl.classList.toggle("val-hot", out.crownTempC >= 240.0);

      const fbVacShow = out.fireboxVacuumPa ?? sp.fireboxVacuumPa ?? 0.30 * out.smokeboxVacuumPa;
      const tiltShow = out.flameTiltDeg ?? sp.flameTiltDeg ?? 0.0;
      document.getElementById("txt-smokebox")!.textContent =
        `煙-${out.smokeboxVacuumPa.toFixed(0)}/火${fbVacShow >= 0 ? "-" : "+"}${Math.abs(fbVacShow).toFixed(0)}Pa | 炎傾${tiltShow.toFixed(0)}°`;
      const gSpdShow = out.groundSpeedKmh ?? out.scaleSpeedKmh;
      const slipShow = out.wheelSlipPct ?? 0.0;
      const spdEl = document.getElementById("txt-speed")!;
      spdEl.textContent = `${out.wheelRpm.toFixed(0)} RPM | ${gSpdShow.toFixed(1)} km/h | 空転${slipShow.toFixed(1)}%`;
      spdEl.classList.toggle("val-hot", slipShow >= 12.0);

      const odoEl = document.getElementById("txt-odometer");
      if (odoEl) {
        const dM = out.distanceM ?? 0.0;
        const sKm = out.scaleDistanceKm ?? (dM * 30.0) / 1000.0;
        const laps = out.lapCount ?? 0.0;
        const pRem = out.predictedRangeM ?? 0.0;
        const p10 = out.predictedRangeP10M ?? 0.0;
        const p90 = out.predictedRangeP90M ?? 0.0;
        const lapStr = laps > 0.01 ? `/${laps.toFixed(1)}周` : "";
        const remStr =
          p10 > 1.0 && p90 > p10
            ? `残${pRem.toFixed(0)}m [${p10.toFixed(0)}..${p90.toFixed(0)}]`
            : pRem > 1.0
              ? `残${pRem.toFixed(0)}m`
              : "残--m";
        odoEl.textContent = `${dM.toFixed(2)}m (${sKm.toFixed(2)}km${lapStr}) | ${remStr}`;
      }

      const flowTags: string[] = [];
      if ((out.waterFillMlS ?? 0) > 0.05) flowTags.push(`注+${out.waterFillMlS.toFixed(1)}`);
      if (out.returnWaterMlS > 0.02) flowTags.push(`戻${out.returnWaterMlS.toFixed(2)}`);
      if ((out.tenderDrainMlS ?? 0) > 0.05) flowTags.push(`排-${out.tenderDrainMlS.toFixed(1)}`);
      const retTag = flowTags.length > 0 ? `${flowTags.join("/")}ml/s` : `${out.tenderWaterTempC.toFixed(0)}°C`;
      document.getElementById("txt-tanks")!.textContent = `${out.alcoholMl.toFixed(0)}ml | ${out.tenderWaterMl.toFixed(0)}ml (${retTag})`;
      const cutPct = out.cutoffPct ?? 0.0;
      document.getElementById("txt-forces")!.textContent =
        `L:${out.forceL.toFixed(1)}N R:${out.forceR.toFixed(1)}N | 締切${cutPct.toFixed(0)}%`;

      const pmpEl = document.getElementById("txt-pump-elastic");
      if (pmpEl) {
        const vacKpa = out.pumpSuctionVacKpa ?? 0.0;
        const pmpP = out.pumpChamberKgf ?? sp.pumpChamberPKgf ?? 0.0;
        const pmpW = out.pumpShaftPowerW ?? 0.0;
        const fBowArr = out.flueDeflectionMm ?? sp.flueDeflectionMm ?? [0];
        const maxBow = Math.max(0, ...fBowArr);
        const pmpStr = vacKpa > 0.2 ? `吸-${vacKpa.toFixed(1)}kPa` : `圧+${Math.max(0, pmpP).toFixed(2)}kgf`;
        pmpEl.textContent = `${pmpStr} (${pmpW.toFixed(2)}W) | 煙管撓${maxBow.toFixed(3)}mm`;
        pmpEl.classList.toggle("val-hot", maxBow > 0.08);
      }

      const gpuBox = document.getElementById("gpu-monitor-box") as HTMLElement | null;
      if (gpuBox) {
        const gm = out.gpuMonitor ?? sp.gpuMonitor;
        if (gpuMonitorEnabled && gm && gm.enabled) {
          gpuBox.style.display = "block";
          gpuBox.innerHTML =
            `<div>GPU: ${gm.device_name || sp.gpuName} (${gm.cfd_hz ?? 125}Hz)</div>` +
            `<div>SM: ${gm.sm_util_pct ?? 0}% | MemBW: ${gm.mem_bw_util_pct ?? 0}% | VRAM: ${gm.vram_used_mb ?? 0}MB</div>` +
            `<div>Temp/Pwr: ${gm.gpu_temp_c ?? 0}°C / ${gm.gpu_power_w ?? 0}W | PDE:${gm.gpu_3d_pde_ms ?? 0}ms PINO:${gm.gpu_pino_ms ?? 0}ms</div>`;
        } else if (gpuMonitorEnabled && !sp.gpuActive) {
          gpuBox.style.display = "block";
          gpuBox.textContent = "GPU Monitor: 离线浏览器 WASM 模式 (请连接本地 8765 GPU 服务)";
        } else {
          gpuBox.style.display = "none";
        }
      }

      const shExitDry = sp.shProfileDryness[sp.shProfileDryness.length - 1] ?? sp.steamDrynessDome;
      document.getElementById("txt-fluid3d")!.textContent =
        `x=${sp.steamDrynessDome.toFixed(2)}→${shExitDry.toFixed(2)} | φ=${sp.airFuelPhi.toFixed(2)} | ${sp.oilFilmUm.toFixed(1)}μm`;
      const etaPct = out.boilerEfficiencyPct ?? sp.boilerEfficiencyPct ?? 0.0;
      const lossW = out.shellHeatLossW ?? sp.shellHeatLossW ?? 0.0;
      const casC = out.casingTempC ?? sp.casingTempC ?? 25.0;
      const wickMlS = out.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0;
      const chtEl = document.getElementById("txt-cht");
      if (chtEl) {
        const qEvap = out.qEvapW ?? 0.0;
        const qFlue = out.qFlueLossW ?? 0.0;
        if (ctrl.whistleValve > 0.05 && (out.whistleSplDb ?? 0) > 10.0) {
          const wf = out.whistleFreqHz ?? 0.0;
          const wx = out.whistleDrynessX ?? 0.95;
          const wtw = out.whistleWallTempC ?? 65.0;
          chtEl.textContent =
            wf > 100
              ? `🔊 11-8汽笛 ${(wf / 1000).toFixed(2)}kHz (${(out.whistleSplDb ?? 0).toFixed(0)}dB·x=${wx.toFixed(2)}·壁${wtw.toFixed(0)}°C)`
              : `🔊 11-8汽笛 閾値未満ヒス (${(out.whistleSplDb ?? 0).toFixed(0)}dB·壁${wtw.toFixed(0)}°C)`;
        } else {
          chtEl.textContent =
            out.burnerLumenConnected === false || sp.burnerLumenConnected === false
              ? `管路不通 0.00ml/s | ${lossW.toFixed(1)}W (${casC.toFixed(0)}°C)`
              : `C型η=${etaPct.toFixed(1)}% (蒸${qEvap.toFixed(0)}W/排煙${qFlue.toFixed(0)}W/殻${lossW.toFixed(0)}W)`;
        }
        chtEl.classList.toggle("val-hot", casC >= 110.0 || wickTipC >= 420.0);
      }
      const sysEl = document.getElementById("txt-sys-clearance");
      if (sysEl) {
        const sprArr = out.springDeflectMm ?? [0.65];
        const sprMean = sprArr.length > 0 ? sprArr.slice(0, 6).reduce((a, b) => a + b, 0) / Math.min(6, sprArr.length) : 0.65;
        const expMm = out.boilerExpansionMm ?? 0.0;
        const vStk = out.lubeSettlingMmS ?? 0.0;
        sysEl.textContent = `懸架撓${sprMean.toFixed(2)}mm | 膨張+${expMm.toFixed(2)}mm | 油水沈降${vStk.toFixed(2)}mm/s`;
      }

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
      if (out.derailed) {
        alarmMsg = `DERAILED! 輪軸脱線・枕木接地制動中 (${out.derailReason || "脱線検知"}) — 冷炉リセットまたは再配置せよ!`;
      } else if (out.burnerLumenConnected === false || sp.burnerLumenConnected === false) {
        alarmMsg = "BURNER LUMEN BLOCKED! 23-1 燃料管〜バーナー芯カップの3D流路遮断 — 毛細吸液不能!";
      } else if (out.hydrolockL || out.hydrolockR) {
        alarmMsg = "HYDROLOCK! 冷気筒水撃ロック発生 — 直ちに8-7室内ドレンレバーを開放せよ!";
      } else if (out.backdraftAlarm || sp.backdraftAlarm) {
        alarmMsg = "BACK-DRAFT! ドラフト喪失・過濃気化による火室逆火発生! 直ちに11-13通風弁を開くか燃料を絞れ!";
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
    if (pipeLabelsEnabled) {
      const snaps = fluidVis.getPipeCallouts();
      const w = window.innerWidth;
      const h = window.innerHeight;
      const proj = new THREE.Vector3();
      const occupied: [number, number][] = [];
      let html = "";
      for (const s of snaps) {
        proj.copy(s.worldPos).project(camera);
        if (proj.z < -0.98 || proj.z > 0.995) continue;
        const sx = (proj.x * 0.5 + 0.5) * w;
        const sy = (-proj.y * 0.5 + 0.5) * h;
        // 避开左右 HUD 面板区域与密集标签互相遮盖
        if (sx < 60 || sx > w - 60 || sy < 56 || sy > h - 28) continue;
        if (sx < 405 && sy < 490) continue;
        if (sx > w - 440 && sy > h - 410) continue;
        let overlap = false;
        for (const [ox, oy] of occupied) {
          if (Math.abs(sx - ox) < 118 && Math.abs(sy - oy) < 24) {
            overlap = true;
            break;
          }
        }
        if (overlap) continue;
        occupied.push([sx, sy]);
        html += `<div class="pipe-callout${s.active ? " flowing" : ""}" style="left:${sx.toFixed(1)}px;top:${sy.toFixed(1)}px;border-left-color:${s.colorCss}">` +
          `<span class="pc-head" style="color:${s.colorCss}">${s.partNo}</span> ${s.title}` +
          `<span class="pc-flow">${s.dirSymbol} ${s.flowText}</span></div>`;
      }
      pipeLabelsLayer.innerHTML = html;
    }
    renderer.render(scene, camera);
  }

  animate();
}

main().catch((err) => {
  loadingText.textContent = `初始化失败: ${err.message}`;
  console.error(err);
});
