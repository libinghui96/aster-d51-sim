// web/src/thermo1d.ts
const PA_PER_KGF_CM2 = 98066.5;
const P_ATM = 101325.0;
const R_STEAM = 461.5;
const GAMMA_STEAM = 1.30;
const CP_WATER = 4186.0;
const CP_COPPER = 385.0;

export interface CabControls {
  fuelValve: number;
  burnerLit: boolean;
  chimneyFan: boolean;
  blowerValve: number;
  regulator: number;
  reverser: number;
  drainCock: boolean;
  bypassValve: number;
  thermalWarp: number;
}

export interface CylinderCalibration {
  qFdcL: number;
  qFdcR: number;
  stroke: number;
  qValveCenterL: number;
  qValveCenterR: number;
  steamLap: number;
  exhaustLap: number;
  portWidth: number;
}

export interface StepOutput {
  forceL: number;
  forceR: number;
  forcePump: number;
  boilerPressureKgf: number;
  steamChestKgf: number;
  superheatTempC: number;
  waterLevelPct: number;
  alcoholMl: number;
  tenderWaterMl: number;
  wheelRpm: number;
  scaleSpeedKmh: number;
  safetyValveOpen: boolean;
  dryFireAlarm: boolean;
  exhaustPlume: number;
  safetyPlume: number;
  drainPlume: number;
}

function satPressurePa(tempC: number): number {
  const tk = Math.max(273.16, Math.min(tempC + 273.15, 640.0));
  const theta = tk / 647.096;
  const tau = 1.0 - theta;
  const lnPi =
    (647.096 / tk) *
    (-7.85951783 * tau +
      1.84408259 * Math.pow(tau, 1.5) -
      11.7866497 * Math.pow(tau, 3.0) +
      22.6807411 * Math.pow(tau, 3.5) -
      15.9618719 * Math.pow(tau, 4.0) +
      1.80122502 * Math.pow(tau, 7.5));
  return 22.064e6 * Math.exp(lnPi);
}

function orificeMassFlow(pUp: number, pDown: number, tempK: number, area: number, cd = 0.68): number {
  if (area <= 1e-12 || pUp <= pDown) return 0.0;
  const dp = pUp - pDown;
  const rCrit = Math.pow(2.0 / (GAMMA_STEAM + 1.0), GAMMA_STEAM / (GAMMA_STEAM - 1.0));
  const pEff = Math.max(pDown, pUp * rCrit);
  const ratio = pEff / pUp;
  const term = Math.max(0.0, Math.pow(ratio, 2.0 / GAMMA_STEAM) - Math.pow(ratio, (GAMMA_STEAM + 1.0) / GAMMA_STEAM));
  const psi = Math.sqrt((2.0 * GAMMA_STEAM / (GAMMA_STEAM - 1.0)) * term);
  const mdot = (cd * area * pUp * psi) / Math.sqrt(R_STEAM * Math.max(290.0, tempK));
  return dp < 1500.0 ? mdot * (dp / 1500.0) : mdot;
}

export class AsterD51Thermo1D {
  readonly A_head = Math.PI * Math.pow(0.015 / 2.0, 2);
  readonly A_crank = (Math.PI * (0.015 * 0.015 - 0.003 * 0.003)) / 4.0;
  readonly A_axlePump = Math.PI * Math.pow(0.0032 / 2.0, 2);
  readonly A_handPump = Math.PI * Math.pow(0.007 / 2.0, 2);
  readonly V_clear: number;
  readonly portBreadth = 0.0055;
  readonly V_boilerTotal = 3.8e-4;
  readonly M_copperShell = 0.46;

  T_boilerC = 25.0;
  M_waterKg = 0.21;
  P_boiler = P_ATM;
  P_steamChest = P_ATM;
  T_superheatC = 25.0;
  safetyValveOpen = false;

  alcoholMl = 130.0;
  sumpFuelMl = 0.0;
  tenderWaterMl = 320.0;
  lubricatorOilPct = 80.0;

  P_cylL = [P_ATM, P_ATM];
  P_cylR = [P_ATM, P_ATM];
  T_cylWallC = 25.0;
  condensateKg = 0.0008;
  private handPumpPendingKg = 0.0;

  constructor(public calib: CylinderCalibration) {
    this.V_clear = 0.12 * this.A_head * calib.stroke;
    this.resetHotReady();
  }

  resetColdPrepared(): void {
    this.T_boilerC = 25.0;
    this.M_waterKg = 0.21;
    this.P_boiler = P_ATM;
    this.P_steamChest = P_ATM;
    this.T_superheatC = 25.0;
    this.safetyValveOpen = false;
    this.alcoholMl = 130.0;
    this.sumpFuelMl = 0.0;
    this.tenderWaterMl = 320.0;
    this.lubricatorOilPct = 80.0;
    this.P_cylL = [P_ATM, P_ATM];
    this.P_cylR = [P_ATM, P_ATM];
    this.T_cylWallC = 25.0;
    this.condensateKg = 0.0008;
  }

  resetHotReady(): void {
    this.resetColdPrepared();
    this.T_boilerC = 149.6;
    this.P_boiler = satPressurePa(this.T_boilerC);
    this.T_superheatC = 165.0;
    this.sumpFuelMl = 3.5;
    this.T_cylWallC = 115.0;
    this.condensateKg = 0.0002;
  }

  triggerHandPump(strokes = 5): void {
    const volMl = strokes * (this.A_handPump * 0.016 * 1e6) * 1.15;
    const used = Math.min(volMl, this.tenderWaterMl);
    this.tenderWaterMl -= used;
    this.handPumpPendingKg += used * 1e-3;
  }

  private valveAreas(qVal: number, qCenter: number): [number, number, number, number] {
    // [已同步修复] 对齐 sim/thermo_1d.py 实测合并杆先导相位: 前进档活塞离开前死点时 xv < -es 打开前腔进气
    const xv = qVal - qCenter;
    const { steamLap: es, exhaustLap: ee, portWidth: wp } = this.calib;
    const b = this.portBreadth;
    return [
      b * Math.min(wp, Math.max(0.0, -xv - es)), // Head-end admission
      b * Math.min(wp, Math.max(0.0, +xv - ee)), // Head-end exhaust
      b * Math.min(wp, Math.max(0.0, +xv - es)), // Crank-end admission
      b * Math.min(wp, Math.max(0.0, -xv - ee)), // Crank-end exhaust
    ];
  }

  private stepCylSide(
    dtSub: number,
    pPair: number[],
    qPiston: number,
    vPiston: number,
    qFdc: number,
    areas: [number, number, number, number],
    drainOpen: boolean
  ): [number, number, number] {
    const [aAdmH, aExhH, aAdmC, aExhC] = areas;
    const tShK = this.T_superheatC + 273.15;
    const aDrain = drainOpen ? 1.1e-6 : 0.0;

    const sP = Math.min(this.calib.stroke, Math.max(0.0, qFdc - qPiston));
    const dsDt = -vPiston;
    const vH = this.V_clear + this.A_head * sP;
    const vC = this.V_clear + this.A_crank * (this.calib.stroke - sP);

    const mInH = orificeMassFlow(this.P_steamChest, pPair[0], tShK, aAdmH);
    const mOutH = orificeMassFlow(pPair[0], P_ATM, tShK, aExhH);
    const mDrnH = orificeMassFlow(pPair[0], P_ATM, tShK, aDrain);
    const dpH =
      ((GAMMA_STEAM * R_STEAM * tShK) / vH) * (mInH - mOutH - mDrnH) -
      ((GAMMA_STEAM * pPair[0]) / vH) * (this.A_head * dsDt);

    const mInC = orificeMassFlow(this.P_steamChest, pPair[1], tShK, aAdmC);
    const mOutC = orificeMassFlow(pPair[1], P_ATM, tShK, aExhC);
    const mDrnC = orificeMassFlow(pPair[1], P_ATM, tShK, aDrain);
    const dpC =
      ((GAMMA_STEAM * R_STEAM * tShK) / vC) * (mInC - mOutC - mDrnC) -
      ((GAMMA_STEAM * pPair[1]) / vC) * (-this.A_crank * dsDt);

    const pRelief = Math.max(this.P_steamChest + 25000.0, P_ATM + 35000.0);
    pPair[0] = Math.min(pRelief, Math.max(0.85 * P_ATM, pPair[0] + dpH * dtSub));
    pPair[1] = Math.min(pRelief, Math.max(0.85 * P_ATM, pPair[1] + dpC * dtSub));

    return [mInH + mInC, mOutH + mOutC, mDrnH + mDrnC];
  }

  step(
    dt: number,
    qPL: number, vPL: number, qVL: number,
    qPR: number, vPR: number, qVR: number,
    omega3: number, vPumpRam: number,
    ctrl: CabControls
  ): StepOutput {
    const dtTh = dt * Math.max(1.0, ctrl.thermalWarp);

    if (ctrl.fuelValve > 0.01 && this.alcoholMl > 0 && this.sumpFuelMl < 3.5) {
      const refill = Math.min(this.alcoholMl, 0.45 * ctrl.fuelValve * dtTh, 3.5 - this.sumpFuelMl);
      this.alcoholMl -= refill;
      this.sumpFuelMl += refill;
    }
    if (this.sumpFuelMl <= 1e-4) ctrl.burnerLit = false;

    const aReg = 3.2e-6 * Math.pow(ctrl.regulator, 1.35);
    const tBoilK = this.T_boilerC + 273.15;
    const hasSteam = this.M_waterKg > 0.015 && this.P_boiler > P_ATM;
    const mdotReg = hasSteam ? orificeMassFlow(this.P_boiler, this.P_steamChest, tBoilK, aReg) : 0.0;

    const areasL = this.valveAreas(qVL, this.calib.qValveCenterL);
    const areasR = this.valveAreas(qVR, this.calib.qValveCenterR);

    const nSub = 4;
    const dtSub = dt / nSub;
    let mdotIn = 0.0, mdotExh = 0.0, mdotDrn = 0.0;

    for (let i = 0; i < nSub; i++) {
      const [inL, exhL, drnL] = this.stepCylSide(dtSub, this.P_cylL, qPL, vPL, this.calib.qFdcL, areasL, ctrl.drainCock);
      const [inR, exhR, drnR] = this.stepCylSide(dtSub, this.P_cylR, qPR, vPR, this.calib.qFdcR, areasR, ctrl.drainCock);
      const inTot = inL + inR;
      const dpSc = ((R_STEAM * (this.T_superheatC + 273.15)) / 2.8e-5) * (mdotReg - inTot);
      this.P_steamChest = Math.min(this.P_boiler, Math.max(P_ATM, this.P_steamChest + dpSc * dtSub));
      mdotIn += inTot / nSub;
      mdotExh += (exhL + exhR) / nSub;
      mdotDrn += (drnL + drnR) / nSub;
    }

    if (mdotIn > 1e-6) {
      this.T_cylWallC += (Math.max(0.0, (this.T_superheatC - this.T_cylWallC) * 1.8) / 42.0) * dtTh;
      if (this.T_cylWallC < 102.0) this.condensateKg = Math.min(0.004, this.condensateKg + 0.25 * mdotIn * dt);
      this.lubricatorOilPct = Math.max(0.0, this.lubricatorOilPct - (80.0 / 3600.0) * dtTh);
    } else {
      this.T_cylWallC += (25.0 - this.T_cylWallC) * 0.002 * dtTh;
    }
    if (ctrl.drainCock && this.condensateKg > 0) {
      this.condensateKg = Math.max(0.0, this.condensateKg - 0.0012 * dtTh);
    }

    const lube = this.lubricatorOilPct > 2.0 ? 1.0 : 2.8;
    const waterDrag = !ctrl.drainCock ? 18.0 * (this.condensateKg / 0.004) : 1.5;
    const calcForce = (p: number[], v: number) =>
      (p[1] - P_ATM) * this.A_crank - (p[0] - P_ATM) * this.A_head - lube * (1.35 * Math.tanh(v * 80.0) + waterDrag * v);

    const forceL = calcForce(this.P_cylL, vPL);
    const forceR = calcForce(this.P_cylR, vPR);

    // [已同步修复] 对齐 ø0.45mm 微孔通风阀面积 (0.16e-6) 与高速射流引风增益 (18000)
    const pGauge = Math.max(0.0, (this.P_boiler - P_ATM) / PA_PER_KGF_CM2);
    if (pGauge >= 4.0) this.safetyValveOpen = true;
    else if (pGauge <= 3.6) this.safetyValveOpen = false;

    const mdotBlw = hasSteam ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, 0.16e-6 * ctrl.blowerValve) : 0.0;
    const mdotSv = hasSteam && this.safetyValveOpen ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, 3.6e-6) : 0.0;

    const draftIdx = (ctrl.chimneyFan ? 1.0 : 0.0) + 18000.0 * mdotBlw + 2600.0 * mdotExh;
    const draftEff = Math.min(1.3, 0.12 + 0.88 * Math.tanh(draftIdx));

    let qComb = 0.0;
    if (ctrl.burnerLit && this.sumpFuelMl > 0) {
      const usedMl = Math.min(this.sumpFuelMl, 0.108 * (0.55 + 0.45 * draftEff) * dtTh);
      this.sumpFuelMl -= usedMl;
      qComb = 0.34 * draftEff * 22.5e6 * ((usedMl / dtTh) * 789e-6);
    }
    const qNet = qComb - 0.42 * (this.T_boilerC - 25.0);
    this.T_superheatC += Math.min(1.0, 1.5 * dtTh) * (this.T_boilerC + (ctrl.burnerLit ? 28.0 * draftEff : 0.0) - this.T_superheatC);

    const vPump = this.tenderWaterMl > 0.5 ? 0.85 * this.A_axlePump * Math.max(0.0, -vPumpRam) : 0.0;
    const dpCheck = Math.max(8000.0, this.P_boiler - P_ATM);
    const vBypCap = 0.65 * (1.6e-7 * Math.pow(ctrl.bypassValve, 1.4)) * Math.sqrt((2.0 * dpCheck) / 1000.0);
    let vToBoil = Math.max(0.0, vPump - vBypCap);
    const usedWaterMl = Math.min(this.tenderWaterMl, vToBoil * 1e6 * dtTh);
    this.tenderWaterMl -= usedWaterMl;
    vToBoil = (usedWaterMl * 1e-6) / dtTh;

    const mdotFw = vToBoil * 1000.0 + (this.handPumpPendingKg > 0 ? this.handPumpPendingKg / dtTh : 0.0);
    this.handPumpPendingKg = 0.0;
    let forcePump = 0.0;
    if (vPumpRam < -1e-4 && this.tenderWaterMl > 0.5) {
      const pLine = vToBoil > 0.0 ? dpCheck : dpCheck * Math.pow(vPump / Math.max(1e-12, vBypCap), 2);
      forcePump = pLine * this.A_axlePump;
    }

    const mdotSteamOut = mdotReg + mdotBlw + mdotSv;
    const hFg = Math.max(1.5e6, (2501.0 - 2.36 * this.T_boilerC) * 1e3);
    const vGas = Math.max(2e-5, this.V_boilerTotal - this.M_waterKg / 960.0);

    if (this.T_boilerC < 99.9) {
      const cTot = this.M_waterKg * CP_WATER + this.M_copperShell * CP_COPPER;
      this.T_boilerC += ((qNet - mdotFw * CP_WATER * (this.T_boilerC - 25.0)) / Math.max(80.0, cTot)) * dtTh;
      this.M_waterKg += mdotFw * dtTh;
      this.P_boiler = P_ATM;
    } else {
      const p1 = satPressurePa(this.T_boilerC);
      const p2 = satPressurePa(this.T_boilerC + 0.1);
      const drhoDt = (p2 / (R_STEAM * (tBoilK + 0.1)) - p1 / (R_STEAM * tBoilK)) / 0.1;
      const cEff = this.M_waterKg * CP_WATER + this.M_copperShell * CP_COPPER + vGas * hFg * drhoDt;
      const dTdt = (qNet - mdotSteamOut * hFg - mdotFw * CP_WATER * (this.T_boilerC - 25.0)) / Math.max(100.0, cEff);
      this.T_boilerC = Math.max(25.0, Math.min(195.0, this.T_boilerC + dTdt * dtTh));
      this.M_waterKg = Math.max(0.0, Math.min(0.35, this.M_waterKg + (mdotFw - mdotSteamOut - vGas * drhoDt * dTdt) * dtTh));
      this.P_boiler = this.M_waterKg > 0.015 ? Math.max(P_ATM, satPressurePa(this.T_boilerC)) : P_ATM;
    }

    const levelPct = Math.max(0.0, Math.min(100.0, ((this.M_waterKg * 1000.0 - 140.0) / 140.0) * 100.0));
    const rpm = (omega3 / (2.0 * Math.PI)) * 60.0;
    const speedKmh = Math.abs(omega3) * (0.5 * 0.0438) * 32.0 * 3.6;

    return {
      forceL,
      forceR,
      forcePump,
      boilerPressureKgf: Math.max(0.0, (this.P_boiler - P_ATM) / PA_PER_KGF_CM2),
      steamChestKgf: Math.max(0.0, (this.P_steamChest - P_ATM) / PA_PER_KGF_CM2),
      superheatTempC: this.T_superheatC,
      waterLevelPct: levelPct,
      alcoholMl: this.alcoholMl + this.sumpFuelMl,
      tenderWaterMl: this.tenderWaterMl,
      wheelRpm: rpm,
      scaleSpeedKmh: speedKmh,
      safetyValveOpen: this.safetyValveOpen,
      dryFireAlarm: this.M_waterKg <= 0.145 && ctrl.burnerLit,
      exhaustPlume: Math.min(1.0, (mdotExh + 0.4 * mdotBlw) * 1800.0),
      safetyPlume: this.safetyValveOpen ? 1.0 : 0.0,
      drainPlume: Math.min(1.0, mdotDrn * 2400.0),
    };
  }
}