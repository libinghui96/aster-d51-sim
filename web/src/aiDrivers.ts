// web/src/aiDrivers.ts
// P2-G 双模 AI 驾驶智能体 (100% 对齐 sim/human_engineer_agent.py 与 sim/ideal_oracle_controller.py)
//   - HumanEngineerTS ("human"): 拟人实操司机 (仅观测 PG-20 量化表压与 11-7 表观水位、0.40s 延迟、0.75s 换手冷却、有限手轮转速、关旁通手摇泵规程)
//   - IdealOracleTS   ("oracle"): 理想全知极限控制 (零延迟读取真实静水位/顶板水深/冷凝水克数、200Hz MIMO 无级膨胀比、3.92 kgf 零泄压巡航)

import { CabControls, StepOutput } from "./thermo1d";

export type AiDriverMode = "off" | "human" | "oracle";

interface DelayedSense {
  simTime: number;
  gaugePKgf: number;
  apparentWaterPct: number;
  speedKmh: number;
  wheelRpm: number;
  safetyPopping: boolean;
  stalled: boolean;
  alcoholEmpty: boolean;
}

function slew(curr: number, target: number, maxRate: number, dt: number): number {
  const diff = target - curr;
  const step = maxRate * dt;
  if (Math.abs(diff) <= step) return target;
  return curr + Math.sign(diff) * step;
}

export class HumanEngineerTS {
  readonly REACTION_DELAY_S = 0.40;
  readonly HAND_SWITCH_COOLDOWN_S = 0.75;
  readonly REVERSER_SLEW = 0.32;
  readonly REGULATOR_SLEW = 0.45;
  readonly BLOWER_SLEW = 0.55;
  readonly BYPASS_SLEW = 0.85;
  readonly HANDPUMP_PERIOD_S = 0.56;

  targetSpeedKmh = 48.0;
  handpumpAnimRad = 0.0;
  statusSummary = "擬人機関士: 運転室計器確認中";

  private senseBuf: DelayedSense[] = [];
  private direction = 1.0;
  private station: "cab" | "tender" = "cab";
  private switchTimer = 0.0;
  private warmupSteamingS = 0.0;
  private handpumpPhase: "idle" | "closing_bypass" | "pumping" | "reopening_bypass" = "idle";
  private pumpStrokeTimer = 0.0;
  private speedErrInt = 0.0;

  reset(direction = 1.0): void {
    this.senseBuf = [];
    this.direction = direction >= 0.0 ? 1.0 : -1.0;
    this.station = "cab";
    this.switchTimer = 0.0;
    this.warmupSteamingS = 0.0;
    this.handpumpPhase = "idle";
    this.pumpStrokeTimer = 0.0;
    this.speedErrInt = 0.0;
    this.handpumpAnimRad = 0.0;
    this.statusSummary = "擬人機関士: 運転室計器確認中";
  }

  private switchStation(target: "cab" | "tender"): boolean {
    if (this.station === target) return this.switchTimer <= 0.0;
    this.station = target;
    this.switchTimer = this.HAND_SWITCH_COOLDOWN_S;
    return false;
  }

  private pushAndReadSense(out: StepOutput, ctrl: CabControls): DelayedSense {
    const quantP = Math.round(out.boilerPressureKgf * 20.0) / 20.0;
    const stalled =
      (ctrl.regulator > 0.25 && out.boilerPressureKgf > 1.2 && Math.abs(out.wheelRpm) < 4.0) ||
      out.hydrolockL ||
      out.hydrolockR;
    const nowSense: DelayedSense = {
      simTime: out.simTime,
      gaugePKgf: quantP,
      apparentWaterPct: Math.round(out.waterLevelPct * 10.0) / 10.0,
      speedKmh: out.scaleSpeedKmh,
      wheelRpm: out.wheelRpm,
      safetyPopping: out.safetyValveOpen,
      stalled,
      alcoholEmpty: out.alcoholMl <= 0.2,
    };
    this.senseBuf.push(nowSense);
    while (this.senseBuf.length > 1 && out.simTime - this.senseBuf[0].simTime >= this.REACTION_DELAY_S) {
      this.senseBuf.shift();
    }
    return this.senseBuf[0];
  }

  update(dt: number, out: StepOutput, ctrl: CabControls, handPumpCb?: (strokes: number) => void): void {
    if (Math.abs(ctrl.reverser) > 0.08) {
      this.direction = ctrl.reverser >= 0.0 ? 1.0 : -1.0;
    }
    const obs = this.pushAndReadSense(out, ctrl);
    if (this.switchTimer > 0.0) {
      this.switchTimer = Math.max(0.0, this.switchTimer - dt);
    }

    ctrl.fuelCapOpen = false;
    ctrl.fireboxDoor = slew(ctrl.fireboxDoor, 0.0, 1.5, dt);

    if (!ctrl.burnerLit && !obs.alcoholEmpty) {
      if (!this.switchStation("tender")) {
        this.statusSummary = "擬人機関士: 炭水車 36-6 燃料弁へ手を移動中...";
        return;
      }
      ctrl.fuelValve = slew(ctrl.fuelValve, 0.88, 0.75, dt);
      if (ctrl.fuelValve >= 0.65) {
        ctrl.burnerLit = true;
        ctrl.chimneyFan = obs.gaugePKgf < 0.9;
        this.statusSummary = "擬人機関士: 23-1 バーナー点火・煙突通風機 ON";
      }
      return;
    }

    if (this.handpumpPhase === "idle" && obs.apparentWaterPct < 38.0 && Math.abs(obs.wheelRpm) < 35.0) {
      this.handpumpPhase = "closing_bypass";
    }

    if (this.handpumpPhase === "closing_bypass") {
      if (!this.switchStation("cab")) {
        this.statusSummary = "擬人機関士: 運転室バイパス弁へ手を戻し中...";
        return;
      }
      ctrl.bypassValve = slew(ctrl.bypassValve, 0.0, this.BYPASS_SLEW, dt);
      this.statusSummary = "擬人機関士: 手動給水前に 4-17 バイパス弁を全閉中";
      if (ctrl.bypassValve <= 0.02) {
        this.handpumpPhase = "pumping";
        this.pumpStrokeTimer = 0.0;
      }
      return;
    }

    if (this.handpumpPhase === "pumping") {
      if (!this.switchStation("tender")) {
        this.statusSummary = "擬人機関士: 炭水車 37-12 手動ポンプ柄へ手を移動中...";
        return;
      }
      this.pumpStrokeTimer += dt;
      this.handpumpAnimRad = 0.42 * Math.sin((2.0 * Math.PI * this.pumpStrokeTimer) / this.HANDPUMP_PERIOD_S);
      if (this.pumpStrokeTimer >= this.HANDPUMP_PERIOD_S) {
        this.pumpStrokeTimer -= this.HANDPUMP_PERIOD_S;
        if (handPumpCb) handPumpCb(1);
      }
      this.statusSummary = `擬人機関士: 31-5 手動ポンプ圧送中 (水面計 ${obs.apparentWaterPct.toFixed(0)}%)`;
      if (obs.apparentWaterPct >= 53.0 || out.tenderWaterMl <= 2.0) {
        this.handpumpAnimRad = 0.0;
        this.handpumpPhase = "reopening_bypass";
      }
      return;
    }

    if (this.handpumpPhase === "reopening_bypass") {
      if (!this.switchStation("cab")) {
        this.statusSummary = "擬人機関士: 炭水車から運転室操作盤へ手を戻し中...";
        return;
      }
      ctrl.bypassValve = slew(ctrl.bypassValve, 0.42, this.BYPASS_SLEW, dt);
      this.statusSummary = "擬人機関士: 手動給水完了、4-17 バイパス弁を復帰中";
      if (ctrl.bypassValve >= 0.40) {
        this.handpumpPhase = "idle";
      }
      return;
    }

    this.handpumpAnimRad = 0.0;
    if (!this.switchStation("cab")) {
      this.statusSummary = "擬人機関士: 運転室へ視線・手を移動中...";
      return;
    }

    let blwTgt: number;
    if (obs.gaugePKgf < 1.0) {
      ctrl.chimneyFan = true;
      blwTgt = 0.05;
    } else {
      ctrl.chimneyFan = false;
      blwTgt = Math.max(0.0, Math.min(0.85, 0.72 * (3.65 - obs.gaugePKgf)));
    }
    if (obs.safetyPopping || obs.gaugePKgf >= 3.90) blwTgt = 0.0;
    ctrl.blowerValve = slew(ctrl.blowerValve, blwTgt, this.BLOWER_SLEW, dt);

    if (obs.gaugePKgf < 1.75 && Math.abs(obs.wheelRpm) < 8.0) {
      ctrl.regulator = slew(ctrl.regulator, 0.0, this.REGULATOR_SLEW, dt);
      ctrl.reverser = slew(ctrl.reverser, this.direction * 0.85, this.REVERSER_SLEW, dt);
      ctrl.drainCock = true;
      this.warmupSteamingS = 0.0;
      this.statusSummary = `擬人機関士: 昇圧待機中 (PG-20 ${obs.gaugePKgf.toFixed(2)} kgf/cm²)`;
      return;
    }

    if (Math.abs(obs.wheelRpm) > 12.0) this.warmupSteamingS += dt;
    if (obs.stalled) {
      this.warmupSteamingS = 0.0;
      ctrl.drainCock = true;
    } else {
      ctrl.drainCock = this.warmupSteamingS < 10.5;
    }

    const revNotch = obs.speedKmh < 18.0 ? 0.85 : obs.speedKmh < 34.0 ? 0.72 : 0.60;
    ctrl.reverser = slew(ctrl.reverser, this.direction * revNotch, this.REVERSER_SLEW, dt);

    const vErr = this.targetSpeedKmh - obs.speedKmh;
    this.speedErrInt = Math.max(-20.0, Math.min(32.0, this.speedErrInt + vErr * dt));
    let regTgt = 0.28 + 0.014 * vErr + 0.0035 * this.speedErrInt;
    if (ctrl.drainCock) regTgt = Math.min(regTgt, 0.42);
    ctrl.regulator = slew(ctrl.regulator, Math.max(0.0, Math.min(0.88, regTgt)), this.REGULATOR_SLEW, dt);

    const wErrApp = obs.apparentWaterPct - 50.0;
    const bypTgt = Math.max(0.0, Math.min(1.0, 0.44 + 0.048 * wErrApp));
    ctrl.bypassValve = slew(ctrl.bypassValve, bypTgt, this.BYPASS_SLEW, dt);

    const actTag = ctrl.drainCock ? "ドレン暖機" : `ノッチ ${Math.round(revNotch * 100)}%`;
    this.statusSummary = `擬人機関士 (遅延0.4s): 加減弁 ${Math.round(ctrl.regulator * 100)}% | ${actTag} | 水面計視認 ${obs.apparentWaterPct.toFixed(0)}%`;
  }
}

export class IdealOracleTS {
  targetSpeedKmh = 48.0;
  pCruiseKgf = 3.92;
  statusSummary = "理想最適制御 (Oracle): 全状態 200Hz MIMO 待機中";

  private direction = 1.0;
  private vInt = 0.0;
  private pumpAccum = 0.0;

  reset(direction = 1.0): void {
    this.direction = direction >= 0.0 ? 1.0 : -1.0;
    this.vInt = 0.0;
    this.pumpAccum = 0.0;
    this.statusSummary = "理想最適制御 (Oracle): 全状態 200Hz MIMO 待機中";
  }

  update(dt: number, out: StepOutput, ctrl: CabControls, handPumpCb?: (strokes: number) => void): void {
    if (Math.abs(ctrl.reverser) > 0.05) {
      this.direction = ctrl.reverser >= 0.0 ? 1.0 : -1.0;
    }
    ctrl.fuelCapOpen = false;
    ctrl.fireboxDoor = 0.0;

    ctrl.burnerLit = out.alcoholMl > 0.1;
    const pErr = this.pCruiseKgf - out.boilerPressureKgf;
    if (out.boilerPressureKgf > 3.94) {
      ctrl.fuelValve = Math.max(0.28, 1.0 - 5.2 * (out.boilerPressureKgf - 3.94));
    } else {
      ctrl.fuelValve = Math.min(1.0, Math.max(0.45, 0.85 + 0.35 * pErr));
    }

    ctrl.chimneyFan = out.boilerPressureKgf < 0.95 && Math.abs(out.wheelRpm) < 15.0;
    const exhFf = Math.min(0.55, 0.32 * out.steamGenGS);
    ctrl.blowerValve = out.boilerPressureKgf >= 3.95 ? 0.0 : Math.max(0.0, Math.min(1.0, 1.15 * pErr - exhFf + 0.08));

    ctrl.drainCock =
      out.hydrolockL ||
      out.hydrolockR ||
      out.condensateG > 0.34 ||
      (out.cylWallTempC < 72.0 && out.condensateG > 0.22);

    const gradeRad = (Math.abs(ctrl.gradeDeg) * Math.PI) / 180.0;
    const trueLvlTarget = Math.min(64.0, 50.0 + 38.0 * Math.tan(gradeRad));
    const wErrTrue = out.trueWaterLevelPct - trueLvlTarget;
    let bypCmd = 0.40 + 0.095 * wErrTrue - 0.24 * out.steamGenGS;
    if (out.crownTempC > 210.0 && out.crownSubmergedMm < 1.2) bypCmd = Math.max(0.40, bypCmd);

    if (wErrTrue < -4.5 && Math.abs(out.wheelRpm) < 28.0 && handPumpCb) {
      ctrl.bypassValve = 0.0;
      this.pumpAccum += dt;
      if (this.pumpAccum >= 0.22) {
        this.pumpAccum = 0.0;
        handPumpCb(1);
      }
    } else {
      this.pumpAccum = 0.0;
      ctrl.bypassValve = Math.max(0.0, Math.min(1.0, bypCmd));
    }

    if (out.boilerPressureKgf < 1.35) {
      ctrl.regulator = 0.0;
      ctrl.reverser = this.direction * 0.88;
      this.statusSummary = `理想最適制御 (Oracle): 最速昇圧中 (${out.boilerTempC.toFixed(1)}°C | ${out.boilerPressureKgf.toFixed(2)} kgf/cm²)`;
      return;
    }

    const spdNorm = Math.max(0.0, Math.min(1.0, out.scaleSpeedKmh / Math.max(12.0, this.targetSpeedKmh)));
    const optCutoff = 0.88 - 0.36 * Math.pow(spdNorm, 0.82);
    ctrl.reverser = this.direction * optCutoff;

    const vErr = this.targetSpeedKmh - out.scaleSpeedKmh;
    this.vInt = Math.max(-15.0, Math.min(30.0, this.vInt + vErr * dt));
    const pComp = 3.60 / Math.max(1.35, out.boilerPressureKgf);
    let regCmd = (0.25 + 0.024 * vErr + 0.0062 * this.vInt) * pComp;
    if (ctrl.drainCock) regCmd = Math.min(regCmd, 0.48);
    ctrl.regulator = Math.max(0.0, Math.min(0.96, regCmd));

    ctrl.blowdownValve = out.trueWaterLevelPct > 78.0 ? 0.35 : 0.0;
    this.statusSummary = `理想最適制御 (Oracle 200Hz): P=${out.boilerPressureKgf.toFixed(2)}kgf | 連続カットオフ ${(optCutoff * 100).toFixed(1)}% | 実水位 ${out.trueWaterLevelPct.toFixed(1)}%`;
  }
}
