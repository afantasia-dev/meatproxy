import os
import math
import time
import random
import multiprocessing as mp
import numpy as np

# =====================================================================
# CARGA DE PYNEC (necpp) CON PROTECCIÓN CONTRA SEGMENTATION FAULT
# =====================================================================
PYNEC_DISPONIBLE = False
try:
    import necpp
    PYNEC_DISPONIBLE = True
except ImportError:
    print("Aviso: Librería 'necpp' (PyNEC) no detectada.")

USAR_PYNEC_EN_BUCLE = False
USAR_MULTICORE = True     # Usa los núcleos del Ryzen dejando 2 libres para no congelar Linux

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y ELÉCTRICOS (BRAZO EVOLUTIVO + PARAGUAS FIJO)
# =====================================================================
RADIO_MAX = 0.040         # 4.0 cm de radio máximo del cilindro (0.04 m)
ALTURA_MAX = 0.070        # 7.0 cm de altura máxima del cilindro (0.07 m)

# Ventana física alcanzable dentro del cilindro R=4cm, H=7cm (6.8 cm a 22.0 cm)
LONGITUD_MIN_M = 0.068    # 6.8 cm longitud mínima estirada del brazo
LONGITUD_MAX_M = 0.220    # 22.0 cm longitud máxima estirada del brazo

RADIO_ALAMBRE = 0.0020    # Radio 2.0 mm (Alambre grueso de 4.0 mm de diámetro)
SEPARACION_AIRE_MM = 4.0  # Espacio libre mínimo contra auto-intersección (mm)
NUM_SEGMENTOS = 10        # 10 pasos 3D progresivos para el brazo único[cite: 1]

# PARÁMETROS FIJOS DEL PARAGUAS / CONO INVERTIDO EN LA BASE (NO EVOLUCIONA)
NUM_RADIALES_PARAGUAS = 4         # 4 radiales simétricos (0°, 90°, 180°, 270°)
LARGO_RADIAL_M = 0.0150           # 1.50 cm (~lambda/4 resonante en 5.50 GHz)
ANGULO_CAIDA_PARAGUAS_DEG = 38.0  # Inclinación hacia abajo (-Z) para 50 Ohm y horizonte XY

FREQ_55 = 5.50e9          # Banda objetivo de optimización (5.50 GHz)
FREQ_24_INFO = 2.44e9     # Solo para referencia en el archivo .nec final
Z0 = 50.0                 # Impedancia puerto RF ESP32-C5 (50 Ohm)[cite: 1]

TAMANO_POBLACION = 512 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 2048
GENERACIONES = 512 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 2048
ELITE_SIZE = 64           # Élite selectiva con filtro anti-clones
TASA_MUTACION = 0.45      # Alta variación para explorar y salir de valores negativos

# Constantes físicas precalculadas
C_LUZ = 299792458.0
EPS_0 = 8.854187817e-12
MU_0 = 4.0 * math.pi * 1e-7
RHO_COBRE = 1.68e-8
SIGMA_COBRE = 5.80e7      # Conductividad del cobre para tarjeta LD de NEC2 (S/m)

K_55 = (2.0 * math.pi * FREQ_55) / C_LUZ
K_EFF_55 = K_55 * 0.88

# =====================================================================
# PRE-CÁLCULO GLOBAL DE MATRICES Y DEL PARAGUAS FIJO (CERO COSTO EN BUCLE)
# =====================================================================
T_VALS_COL = np.linspace(0.0, 1.0, 10, dtype=float)[np.newaxis, :, np.newaxis]
IDX_COL_I, IDX_COL_J = np.triu_indices(NUM_SEGMENTOS, k=2)
DIAMETRO_ALAMBRE_M = 2.0 * RADIO_ALAMBRE
DIST_MIN_REQ_M = DIAMETRO_ALAMBRE_M + (SEPARACION_AIRE_MM / 1000.0)

def _construir_geometria_paraguas_fijo():
    """
    Genera los nodos fijos de los radiales del cono invertido que nacen en (0,0,0)
    y caen hacia -Z[cite: 1]. No evolucionan.
    """
    alpha_rad = math.radians(ANGULO_CAIDA_PARAGUAS_DEG)
    r_horiz = LARGO_RADIAL_M * math.cos(alpha_rad)
    z_punta = -LARGO_RADIAL_M * math.sin(alpha_rad)

    radiales = []
    for k in range(NUM_RADIALES_PARAGUAS):
        phi = k * (2.0 * math.pi / NUM_RADIALES_PARAGUAS)
        p_inicio = np.array([0.0, 0.0, 0.0], dtype=float)
        p_fin = np.array([r_horiz * math.cos(phi), r_horiz * math.sin(phi), z_punta], dtype=float)
        radiales.append((p_inicio, p_fin))
    return radiales

RADIALES_PARAGUAS_FIJOS = _construir_geometria_paraguas_fijo()

# Puntos discretos de los radiales fijos para checar que el brazo no los toque
PTS_PARAGUAS_COL = np.vstack([
    np.array([p0 + t * (p1 - p0) for t in np.linspace(0.25, 1.0, 6)])
    for (p0, p1) in RADIALES_PARAGUAS_FIJOS
])

def _construir_matrices_direcciones_y_paraguas():
    phis = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    thetas_esf = np.linspace(0.12, math.pi - 0.12, 7)

    dirs_esfera = []
    pesos_esfera = []
    for th in thetas_esf:
        s_th, c_th = math.sin(th), math.cos(th)
        for ph in phis:
            dirs_esfera.append([s_th * math.cos(ph), s_th * math.sin(ph), c_th])
            pesos_esfera.append(s_th)

    pesos_arr = np.array(pesos_esfera, dtype=float)
    pesos_arr /= np.sum(pesos_arr)

    # Domo +Z: 3 anillos (85°, 60°, 30°) de 12 puntos + cúspide +Z (0°) = 37 puntos
    dirs_domo = []
    for deg in (85.0, 60.0, 30.0):
        th = math.radians(deg)
        s_th, c_th = math.sin(th), math.cos(th)
        for ph in phis:
            dirs_domo.append([s_th * math.cos(ph), s_th * math.sin(ph), c_th])
    dirs_domo.append([0.0, 0.0, 1.0])

    r_hat_all = np.vstack([np.array(dirs_esfera, dtype=float),
                           np.array(dirs_domo, dtype=float)])

    # Precalcular el campo eléctrico base aportado por los radiales del paraguas fijo
    vecs_rad = []
    centros_rad = []
    weights_rad = []
    for (p0, p1) in RADIALES_PARAGUAS_FIJOS:
        v = p1 - p0
        l_r = float(np.linalg.norm(v))
        u_r = v / l_r
        c_r = 0.5 * (p0 + p1)
        # Corriente de retorno dividida entre los N radiales (-1 / N)
        i_w = -(l_r / NUM_RADIALES_PARAGUAS) * math.sin(K_EFF_55 * (0.5 * l_r))
        vecs_rad.append(u_r)
        centros_rad.append(c_r)
        weights_rad.append(i_w)

    vecs_rad = np.array(vecs_rad, dtype=float)
    centros_rad = np.array(centros_rad, dtype=float)
    weights_rad = np.array(weights_rad, dtype=complex)

    fases_rad = K_55 * (r_hat_all @ centros_rad.T)
    W_rad = (np.cos(fases_rad) + 1j * np.sin(fases_rad)) * weights_rad[np.newaxis, :]
    J_rad = W_rad @ vecs_rad
    dot_jr_rad = np.sum(J_rad * r_hat_all, axis=1, keepdims=True)
    e_paraguas_base = J_rad - r_hat_all * dot_jr_rad

    # Factor reflector cónico del paraguas (empuja radiación de -Z hacia +Z y el horizonte XY)
    z_cos = r_hat_all[:, 2]
    factor_reflector = np.where(z_cos >= -0.05, 1.18 + 0.15 * np.maximum(z_cos, 0.0), 0.72)

    return r_hat_all, pesos_arr, e_paraguas_base, factor_reflector

R_HAT_ALL, PESOS_ESFERA, E_PARAGUAS_BASE, FACTOR_REFLECTOR_CONO = _construir_matrices_direcciones_y_paraguas()

# =====================================================================
# 2. CODIFICACIÓN DE HÉLICE PROGRESIVA (SOLO EVOLUCIONA EL BRAZO)
# =====================================================================
def crear_individuo():
    genes = []
    dz_medio = ALTURA_MAX / NUM_SEGMENTOS
    for _ in range(NUM_SEGMENTOS):
        r = random.uniform(0.008, RADIO_MAX)
        d_theta = random.uniform(0.25, 1.35)
        d_z = random.uniform(dz_medio * 0.5, dz_medio * 1.45)
        genes.append([r, d_theta, d_z])
    return np.array(genes, dtype=float)

def copiar_individuo(ind):
    return np.copy(ind)

def decodificar_alambre_a_cartesianas(genes):
    """
    Construye el único brazo activo paso a paso desde el origen (0,0,0) hacia +Z[cite: 1].
    """
    nodos = np.zeros((NUM_SEGMENTOS + 1, 3), dtype=float)
    dz_clamped = np.maximum(genes[:, 2], 0.0025)
    suma_dz = float(np.sum(dz_clamped))
    escala_z = min(1.0, ALTURA_MAX / max(suma_dz, 1e-6))

    theta_actual = 0.0
    z_actual = 0.0
    r_prev = RADIO_ALAMBRE * 3.5

    for idx in range(NUM_SEGMENTOS):
        r, d_theta, d_z = genes[idx]
        r_target = min(max(float(r), RADIO_ALAMBRE * 3.0), RADIO_MAX)
        r_clip = 0.15 * r_prev + 0.85 * r_target
        r_prev = r_clip

        theta_actual += float(d_theta)
        paso_z = max(float(d_z) * escala_z, 0.0035)
        z_actual = min(ALTURA_MAX, z_actual + paso_z)

        nodos[idx + 1, 0] = r_clip * math.cos(theta_actual)
        nodos[idx + 1, 1] = r_clip * math.sin(theta_actual)
        nodos[idx + 1, 2] = z_actual

    return nodos

# =====================================================================
# 3. AUTO-COLISIÓN Y MODELO 3D VECTORIAL CON PARAGUAS FIJO
# =====================================================================
def motor_autocolision_alambre(nodos):
    p1 = nodos[:-1]
    vecs = nodos[1:] - p1
    l_tramos = np.linalg.norm(vecs, axis=1)

    penalizacion = 0.0
    cortos = np.count_nonzero(l_tramos < DIAMETRO_ALAMBRE_M * 1.5)
    if cortos > 0:
        penalizacion += cortos * 5000.0

    # Auto-colisión entre los 10 segmentos del brazo
    pts = p1[:, np.newaxis, :] + T_VALS_COL * vecs[:, np.newaxis, :]
    diff = pts[IDX_COL_I, :, np.newaxis, :] - pts[IDX_COL_J, np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    d_ij = np.sqrt(np.min(dists_sq, axis=(1, 2)))

    min_dist_centros = float(np.min(d_ij))
    mask_col = d_ij < DIST_MIN_REQ_M
    if np.any(mask_col):
        violaciones = DIST_MIN_REQ_M - d_ij[mask_col]
        penalizacion += float(np.sum(violaciones * 80000.0 + 300.0))

    # Verificar que los segmentos superiores del brazo no choquen con el paraguas fijo
    pts_brazo_sup = pts[1:].reshape(-1, 3)
    diff_paraguas = pts_brazo_sup[:, np.newaxis, :] - PTS_PARAGUAS_COL[np.newaxis, :, :]
    min_dist_paraguas = float(np.sqrt(np.min(np.sum(diff_paraguas * diff_paraguas, axis=-1))))
    if min_dist_paraguas < DIST_MIN_REQ_M:
        penalizacion += (DIST_MIN_REQ_M - min_dist_paraguas) * 80000.0 + 300.0

    min_dist_centros = min(min_dist_centros, min_dist_paraguas)
    aire_libre_mm = (min_dist_centros - DIAMETRO_ALAMBRE_M) * 1000.0
    return penalizacion, aire_libre_mm, vecs, l_tramos

def evaluar_patron_3d_55ghz_sin_negativos(nodos, vecs, longitudes):
    """
    Evalúa los 121 ángulos sumando el campo del brazo evolucionado + el campo precalculado
    del paraguas cónico invertido fijo.
    """
    penalizacion_geom = 0.0
    longitudes_seguras = np.maximum(longitudes, 1e-5)
    vectores = vecs / longitudes_seguras[:, np.newaxis]
    centros = 0.5 * (nodos[:-1] + nodos[1:])

    longitud_total = float(np.sum(longitudes))
    pos_acumulada = np.cumsum(longitudes) - 0.5 * longitudes

    if longitud_total < LONGITUD_MIN_M:
        penalizacion_geom += (LONGITUD_MIN_M - longitud_total) * 1500.0
    elif longitud_total > LONGITUD_MAX_M:
        penalizacion_geom += (longitud_total - LONGITUD_MAX_M) * 1500.0

    cos_angs = np.sum(vectores[:-1] * vectores[1:], axis=1)
    mask_agudos = cos_angs < 0.20
    if np.any(mask_agudos):
        penalizacion_geom += float(np.sum((0.20 - cos_angs[mask_agudos]) * 30.0))

    centro_masa_xy = np.mean(nodos[1:, :2], axis=0)
    penalizacion_geom += float(np.linalg.norm(centro_masa_xy)) * 550.0

    # Corriente de onda estacionaria en los 10 segmentos del brazo
    dist_punta = longitud_total - pos_acumulada
    k_dist = K_EFF_55 * dist_punta
    i_weight = longitudes * (np.sin(k_dist) + 0.25j * np.cos(k_dist))

    # Corriente en la base (0,0,0) que alimenta en contrafase al cono invertido fijo
    i_base = math.sin(K_EFF_55 * longitud_total) + 0.25j * math.cos(K_EFF_55 * longitud_total)

    fases = K_55 * (R_HAT_ALL @ centros.T)
    W = (np.cos(fases) + 1j * np.sin(fases)) * i_weight[np.newaxis, :]
    J_vec = W @ vectores
    dot_jr = np.sum(J_vec * R_HAT_ALL, axis=1, keepdims=True)
    E_brazo = J_vec - R_HAT_ALL * dot_jr

    # Suma vectorial del brazo + paraguas fijo y efecto reflector del cono invertido
    E_total = (E_brazo + i_base * E_PARAGUAS_BASE) * FACTOR_REFLECTOR_CONO[:, np.newaxis]
    pot_all = np.sum(E_total.real ** 2 + E_total.imag ** 2, axis=1) + 1e-12

    # Paso A: Potencia media isotrópica (P_iso = 0 dBi)
    p_iso = max(float(np.dot(pot_all[:84], PESOS_ESFERA)), 1e-12)

    # Paso B: Ganancia en dBi en los 37 puntos del domo +Z
    dbi_domo = 10.0 * np.log10(pot_all[84:] / p_iso)
    dbi_anillos = dbi_domo[:36].reshape(3, 12)  # Filas: 85°, 60°, 30°
    dbi_z_top = float(dbi_domo[36])

    ripples = (np.max(dbi_anillos, axis=1) - np.min(dbi_anillos, axis=1)) + 2.0 * np.std(dbi_anillos, axis=1)
    rizado_xy_55 = float(np.mean(ripples))

    g_min_z_pos = float(np.min(dbi_domo))
    g_max_z_pos = float(np.max(dbi_domo))
    spread_3d_55 = g_max_z_pos - g_min_z_pos

    # Paso C: Penalización fuerte vectorizada contra valores negativos en +Z (dBi < 0)
    penalizacion_negativos_z = 0.0
    mask_neg = dbi_domo < 0.0
    if np.any(mask_neg):
        deficits = np.abs(dbi_domo[mask_neg])
        penalizacion_negativos_z += float(np.sum(deficits * 18.0 + (deficits ** 2) * 8.0))
        peor_deficit = abs(g_min_z_pos)
        penalizacion_negativos_z += (peor_deficit * 90.0) + ((peor_deficit ** 2) * 45.0)
    else:
        penalizacion_negativos_z -= (g_min_z_pos * 25.0)

    media_xy = float(np.mean(dbi_anillos[0]))
    deficit_z_55 = abs(dbi_z_top - media_xy) * 1.5

    return spread_3d_55, rizado_xy_55, deficit_z_55, penalizacion_geom, penalizacion_negativos_z, g_min_z_pos, g_max_z_pos, longitud_total

# =====================================================================
# 4. MODELO DE IMPEDANCIA VECTORIZADO (BRAZO SOBRE PARAGUAS A 38°)
# =====================================================================
def calcular_z_in_un_alambre(nodos, vecs, longitudes, freq):
    """
    Calcula la escalera RLC considerando el plano de tierra cónico invertido (38°)
    que estabiliza la resistencia de radiación cerca de 50 Ohm[cite: 1].
    """
    omega = 2.0 * math.pi * freq
    longitud_onda = C_LUZ / freq
    delta_skin = math.sqrt(RHO_COBRE / (math.pi * freq * MU_0))

    l_seg = np.maximum(longitudes, 1e-4)
    longitud_total = float(np.sum(l_seg))
    z_medio = np.maximum(0.5 * (nodos[:-1, 2] + nodos[1:, 2]), 0.001)
    cos_alpha = np.abs(vecs[:, 2]) / l_seg

    term_log_l = np.maximum(np.log((2.0 * l_seg) / RADIO_ALAMBRE) - 1.0, 0.15)
    L_segs = (MU_0 / (2.0 * math.pi)) * l_seg * term_log_l * 0.40

    # El cono de radiales aumenta el acoplamiento capacitivo cerca de la base
    term_log_c = np.maximum(np.log((1.8 * z_medio) / RADIO_ALAMBRE), 0.22)
    C_segs = ((2.0 * math.pi * EPS_0 * l_seg) / term_log_c) * 0.72

    R_skin = (RHO_COBRE * l_seg) / (2.0 * math.pi * RADIO_ALAMBRE * delta_skin)
    l_norm = np.minimum(l_seg / longitud_onda, 0.5)
    R_rad = 73.0 * (math.pi ** 2) * (l_norm ** 2) * (0.5 + 0.5 * (cos_alpha ** 2))
    R_segs = np.maximum(R_rad + R_skin, 0.02)

    beta = (2.0 * math.pi) / longitud_onda
    elec_len = beta * longitud_total * 0.82
    x_apertura = min(max((-Z0 / (math.tan(elec_len) + 1e-6)) * 0.28, -250.0), 250.0)
    # Los radiales inclinados a ~38° elevan la resistencia base natural hacia ~48-52 Ohm
    r_apertura = 44.0 + 8.0 * abs(math.sin(elec_len))

    z_carga = complex(r_apertura, x_apertura)
    for idx in range(NUM_SEGMENTOS - 1, -1, -1):
        z_serie = complex(float(R_segs[idx]), omega * float(L_segs[idx]))
        y_paralelo = complex(0.0, omega * float(C_segs[idx]))
        z_carga = z_serie + (1.0 / ((1.0 / (z_carga + 1e-12)) + y_paralelo))

    return z_carga

# =====================================================================
# 5. SIMULADOR PYNEC (necpp) + EXPORTADORES .NEC Y .SCAD (CON PARAGUAS)
# =====================================================================
def calcular_subsegmentos_seguros(l_tramo):
    longitud_min_subseg = RADIO_ALAMBRE * 2.2
    n_max = int(l_tramo // longitud_min_subseg)
    return max(1, min(5, n_max))

def simular_un_alambre_pynec(nodos_m, freq_hz):
    if not PYNEC_DISPONIBLE:
        return None

    for i in range(len(nodos_m) - 1):
        l_tramo = float(np.linalg.norm(nodos_m[i+1] - nodos_m[i]))
        if l_tramo < RADIO_ALAMBRE * 2.1:
            return None

    nec = necpp.nec_create()
    try:
        # 1. Alambres del brazo evolucionado (Tags 1 a NUM_SEGMENTOS)
        for i in range(len(nodos_m) - 1):
            p1, p2 = nodos_m[i], nodos_m[i+1]
            l_tramo = float(np.linalg.norm(p2 - p1))
            n_segs = calcular_subsegmentos_seguros(l_tramo)
            necpp.nec_wire(nec, i + 1, n_segs,
                           float(p1[0]), float(p1[1]), float(p1[2]),
                           float(p2[0]), float(p2[1]), float(p2[2]),
                           RADIO_ALAMBRE, 1.0, 1.0)

        # 2. Alambres de los radiales del paraguas fijo (Tags 101 a 104) unidos en (0,0,0)
        n_segs_rad = calcular_subsegmentos_seguros(LARGO_RADIAL_M)
        for idx_r, (p0, p1) in enumerate(RADIALES_PARAGUAS_FIJOS):
            necpp.nec_wire(nec, 101 + idx_r, n_segs_rad,
                           float(p0[0]), float(p0[1]), float(p0[2]),
                           float(p1[0]), float(p1[1]), float(p1[2]),
                           RADIO_ALAMBRE, 1.0, 1.0)

        necpp.nec_geometry_complete(nec, 0)
        necpp.nec_ld_card(nec, 5, 0, 0, 0, SIGMA_COBRE, 0.0, 0.0)
        necpp.nec_fr_card(nec, 0, 1, freq_hz / 1e6, 0.0)
        # Excitación en el primer segmento del brazo (Tag 1, Seg 1) justo sobre el cono
        necpp.nec_ex_card(nec, 0, 1, 1, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        necpp.nec_rp_card(nec, 0, 10, 37, 0, 5, 0, 0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0)

        z_real = float(necpp.nec_impedance_real(nec, 0))
        z_imag = float(necpp.nec_impedance_imag(nec, 0))
        z_in_nec = complex(z_real, z_imag)

        gamma = abs((z_in_nec - Z0) / (z_in_nec + Z0))
        s11_nec = 20.0 * math.log10(max(gamma, 1e-5))

        gain_max = float(necpp.nec_gain_max(nec, 0))
        gain_min = float(necpp.nec_gain_min(nec, 0))
        gain_sd = float(necpp.nec_gain_sd(nec, 0))
        delta_dbi = gain_max - max(gain_min, -40.0)

        necpp.nec_delete(nec)
        return z_in_nec, s11_nec, gain_max, gain_min, delta_dbi, gain_sd
    except Exception:
        try:
            necpp.nec_delete(nec)
        except Exception:
            pass
        return None

def generar_archivo_nec_campeon(nodos_m, filename="antena_paraguas_55ghz_esp32c5.nec"):
    lineas_nec = [
        "CM ===================================================================",
        "CM ANTENA 5.50 GHz: 1 BRAZO EVOLUCIONADO + PARAGUAS CONO INVERTIDO FIJO",
        f"CM Cilindro Brazo: R={RADIO_MAX*100:.1f} cm | H={ALTURA_MAX*100:.1f} cm | Alambre D={RADIO_ALAMBRE*2000:.1f} mm",
        f"CM Paraguas Base: {NUM_RADIALES_PARAGUAS} radiales de L={LARGO_RADIAL_M*100:.2f} cm a -{ANGULO_CAIDA_PARAGUAS_DEG:.1f} deg",
        "CM Unidades NEC: Metros (m)",
        "CE"
    ]

    # 1. Tarjetas GW del brazo evolucionado
    for i in range(len(nodos_m) - 1):
        p1, p2 = nodos_m[i], nodos_m[i+1]
        tag = i + 1
        l_tramo = float(np.linalg.norm(p2 - p1))
        n_segs = calcular_subsegmentos_seguros(l_tramo)
        lineas_nec.append(
            f"GW {tag:3d} {n_segs} {p1[0]:9.6f} {p1[1]:9.6f} {p1[2]:9.6f} "
            f"{p2[0]:9.6f} {p2[1]:9.6f} {p2[2]:9.6f} {RADIO_ALAMBRE:8.5f}"
        )

    # 2. Tarjetas GW del Paraguas / Cono Invertido fijo en la base (Tags 101..104)
    n_segs_rad = calcular_subsegmentos_seguros(LARGO_RADIAL_M)
    for idx_r, (p0, p1) in enumerate(RADIALES_PARAGUAS_FIJOS):
        tag_r = 101 + idx_r
        lineas_nec.append(
            f"GW {tag_r:3d} {n_segs_rad} {p0[0]:9.6f} {p0[1]:9.6f} {p0[2]:9.6f} "
            f"{p1[0]:9.6f} {p1[1]:9.6f} {p1[2]:9.6f} {RADIO_ALAMBRE:8.5f}"
        )

    lineas_nec.extend([
        "GE 0",
        f"LD 5 0 0 0 {SIGMA_COBRE:.2e}",
        "EX 0 1 1 0 1.0 0.0",
        f"FR 0 1 0 0 {FREQ_55/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        f"FR 0 1 0 0 {FREQ_24_INFO/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        "EN"
    ])

    contenido = "\n".join(lineas_nec)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(contenido + "\n")
    except Exception as e:
        print(f"Aviso al guardar archivo .nec: {e}")
    return contenido

def generar_archivo_scad_campeon(nodos_cm, filename="antena_paraguas_55ghz_esp32c5.scad"):
    """
    Genera y guarda el script completo de OpenSCAD (.scad) en milímetros (mm) o centímetros (cm)
    incluyendo el brazo evolucionado y el cono invertido (paraguas) en la base.
    """
    nodos_mm = nodos_cm * 10.0  # Convertimos a milímetros para impresión 3D / visualización CAD directa
    radiales_mm = [(p0 * 1000.0, p1 * 1000.0) for (p0, p1) in RADIALES_PARAGUAS_FIJOS]

    lineas_scad = [
        "// ===================================================================",
        "// OPENSCAD: ANTENA 5.50 GHz (1 BRAZO EVOLUCIONADO + PARAGUAS FIJO)",
        "// Unidades: Milímetros (mm)",
        "// ===================================================================",
        "$fn = 36;",
        f"radio_alambre_mm = {RADIO_ALAMBRE * 1000.0:.2f};",
        "",
        "// 1. Nodos del Brazo Único Evolucionado (en mm, creciendo hacia +Z)",
        "nodos_brazo_mm = ["
    ]

    for idx, pt in enumerate(nodos_mm):
        coma = "," if idx < len(nodos_mm) - 1 else ""
        lineas_scad.append(f"    [{pt[0]:8.3f}, {pt[1]:8.3f}, {pt[2]:8.3f}]{coma} // Paso {idx}")
    lineas_scad.append("];\n")

    lineas_scad.append("// 2. Radiales del Paraguas / Cono Invertido Fijo (en mm, hacia -Z)")
    lineas_scad.append("radiales_paraguas_mm = [")
    for idx_r, (p0, p1) in enumerate(radiales_mm):
        coma = "," if idx_r < len(radiales_mm) - 1 else ""
        lineas_scad.append(
            f"    [[{p0[0]:7.3f}, {p0[1]:7.3f}, {p0[2]:7.3f}], "
            f"[{p1[0]:7.3f}, {p1[1]:7.3f}, {p1[2]:7.3f}]]{coma} // Radial {idx_r + 1}"
        )
    lineas_scad.extend([
        "];",
        "",
        "// Módulo para construir el brazo continuo con uniones esféricas suaves",
        "module construir_brazo(pts, r) {",
        "    for (i = [0 : len(pts) - 2]) {",
        "        hull() {",
        "            translate(pts[i]) sphere(r = r);",
        "            translate(pts[i+1]) sphere(r = r);",
        "        }",
        "    }",
        "}",
        "",
        "// Módulo para construir el Paraguas / Cono Invertido en la base (0,0,0)",
        "module construir_paraguas(rads, r) {",
        "    for ( k = [0 : len(rads) - 1] ) {",
        "        hull() {",
        "            translate(rads[k][0]) sphere(r = r);",
        "            translate(rads[k][1]) sphere(r = r);",
        "        }",
        "    }",
        "}",
        "",
        "// Renderizado del conjunto completo",
        "union() {",
        "    color(\"DarkOrange\") construir_brazo(nodos_brazo_mm, radio_alambre_mm);",
        "    color(\"DodgerBlue\") construir_paraguas(radiales_paraguas_mm, radio_alambre_mm);",
        "}"
    ])

    contenido_scad = "\n".join(lineas_scad)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(contenido_scad + "\n")
    except Exception as e:
        print(f"Aviso al guardar archivo .scad: {e}")
    return contenido_scad

# =====================================================================
# 6. FUNCIÓN DE APTITUD (LISTA PARA MULTIPROCESAMIENTO)
# =====================================================================
def evaluar_individuo(individuo):
    nodos = decodificar_alambre_a_cartesianas(individuo)

    pen_col, aire_mm, vecs, l_tramos = motor_autocolision_alambre(nodos)
    (spread_3d_55, ripple_xy_55, deficit_z_55,
     pen_geom, pen_negativos_z, g_min_z, g_max_z, len_total_m) = evaluar_patron_3d_55ghz_sin_negativos(nodos, vecs, l_tramos)

    if PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE and pen_col == 0.0:
        res55 = simular_un_alambre_pynec(nodos, FREQ_55)
    else:
        res55 = None

    if res55 is not None:
        z_in_55, s11_55, gmax55, gmin55, delta_dbi_55, sd55 = res55
        varianza_dbi_55 = delta_dbi_55 + 2.5 * sd55 + ripple_xy_55
        g_min_z = gmin55
        g_max_z = gmax55
        if gmin55 < 0.0:
            pen_negativos_z += (abs(gmin55) * 110.0) + ((abs(gmin55) ** 2) * 55.0)
    else:
        z_in_55 = calcular_z_in_un_alambre(nodos, vecs, l_tramos, FREQ_55)
        gamma_55 = abs((z_in_55 - Z0) / (z_in_55 + Z0))
        s11_55 = 20.0 * math.log10(max(gamma_55, 1e-5))
        varianza_dbi_55 = spread_3d_55 + (ripple_xy_55 * 1.8) + deficit_z_55

    penalizacion_s11 = 0.0
    if s11_55 > -12.0:
        penalizacion_s11 += (s11_55 - (-12.0)) * 12.0

    s11_efectivo = max(s11_55, -45.0)
    fitness = (-s11_efectivo * 2.8) - (varianza_dbi_55 * 4.0) - pen_negativos_z - penalizacion_s11 - pen_geom - pen_col

    return (fitness, s11_55, z_in_55, spread_3d_55, ripple_xy_55, g_min_z, g_max_z, aire_mm, len_total_m, individuo)

# =====================================================================
# 7. SELECCIÓN DE ÉLITE CON CACHÉ DE TUPLAS (EVITA RE-EVALUAR LA ÉLITE)
# =====================================================================
def seleccionar_elite_sin_negativos(evaluados, elite_size):
    vip_positivos = []
    resto = []

    for item in evaluados:
        if item[5] >= 0.0 and item[7] >= SEPARACION_AIRE_MM:
            vip_positivos.append(item)
        else:
            resto.append(item)

    pool_ordenado = vip_positivos + resto

    elite_tuplas = []
    distancia_min_genes = 0.015

    for candidato in pool_ordenado:
        ind_cand = candidato[9]
        es_clon = False
        for miembro in elite_tuplas:
            if float(np.linalg.norm(ind_cand - miembro[9])) < distancia_min_genes:
                es_clon = True
                break
        if not es_clon:
            elite_tuplas.append(candidato)
        if len(elite_tuplas) >= elite_size:
            break

    idx = 0
    while len(elite_tuplas) < elite_size and idx < len(pool_ordenado):
        elite_tuplas.append(pool_ordenado[idx])
        idx += 1

    return elite_tuplas

def cruzar_individuos(padre1, padre2):
    mask = np.random.rand(NUM_SEGMENTOS, 1) < 0.5
    return np.where(mask, padre1, padre2)

def mutar_individuo(individuo):
    mutado = copiar_individuo(individuo)
    dz_max_paso = (ALTURA_MAX / NUM_SEGMENTOS) * 1.8
    for i in range(NUM_SEGMENTOS):
        if random.random() < TASA_MUTACION:
            escala = 3.0 if random.random() < 0.15 else 1.0
            mutado[i, 0] = np.clip(mutado[i, 0] + random.gauss(0, 0.0055 * escala), RADIO_ALAMBRE * 3.0, RADIO_MAX)
            mutado[i, 1] = np.clip(mutado[i, 1] + random.gauss(0, 0.30 * escala), -0.35, 1.65)
            mutado[i, 2] = np.clip(mutado[i, 2] + random.gauss(0, 0.0035 * escala), 0.0035, dz_max_paso)
    return mutado

def ejecutar_optimizacion():
    t_inicio = time.time()
    num_workers = max(1, (os.cpu_count() or 4) - 2) if USAR_MULTICORE else 1

    print("=" * 110)
    print(" OPTIMIZACIÓN 5.50 GHz: 1 BRAZO EVOLUTIVO + PARAGUAS CONO INVERTIDO FIJO EN LA BASE")
    print(f" Cilindro Brazo: R={RADIO_MAX*100:.1f} cm, H={ALTURA_MAX*100:.1f} cm | Paraguas Fijo: {NUM_RADIALES_PARAGUAS} radiales de {LARGO_RADIAL_M*100:.2f} cm a -{ANGULO_CAIDA_PARAGUAS_DEG:.0f}°")
    print(f" Hilos activos: {num_workers} (2 libres para el sistema) | Población: {TAMANO_POBLACION} | Generaciones: {GENERACIONES}")
    print("=" * 110)

    poblacion_inicial = [crear_individuo() for _ in range(TAMANO_POBLACION)]
    elite_tuplas_cache = []

    pool = mp.Pool(processes=num_workers) if num_workers > 1 else None

    try:
        nuevos_individuos = poblacion_inicial
        for gen in range(GENERACIONES):
            if pool is not None:
                evaluados_nuevos = pool.map(evaluar_individuo, nuevos_individuos, chunksize=64)
            else:
                evaluados_nuevos = [evaluar_individuo(ind) for ind in nuevos_individuos]

            evaluados = elite_tuplas_cache + evaluados_nuevos
            evaluados.sort(key=lambda x: x[0], reverse=True)

            if gen % 32 == 0 or gen == GENERACIONES - 1:
                bfit, bs55, bz55, bsp, brip, bgmin, bgmax, baire, blen, mejor_ind = evaluados[0]
                info_nec = ""
                if PYNEC_DISPONIBLE and baire > 1.0:
                    nodos_g = decodificar_alambre_a_cartesianas(mejor_ind)
                    res_nec55 = simular_un_alambre_pynec(nodos_g, FREQ_55)
                    if res_nec55 is not None:
                        _, _, gmax55, gmin55, _, _ = res_nec55
                        info_nec = f" | NEC_Z+:[{gmin55:+4.1f},{gmax55:+4.1f}]dBi"

                t_trans = time.time() - t_inicio
                print(f"Gen {gen:4d}/{GENERACIONES} ({t_trans:5.1f}s) | Fit: {bfit:7.2f} | "
                      f"S11: {bs55:5.1f}dB | Domo_Z+:[{bgmin:+5.2f} a {bgmax:+5.2f}]dBi | "
                      f"Sp3D: {bsp:4.2f}dB | RipXY: {brip:4.2f}dB | L: {blen*100:5.2f}cm{info_nec}")

            elite_tuplas_cache = seleccionar_elite_sin_negativos(evaluados, ELITE_SIZE)
            mejores_genes = [item[9] for item in elite_tuplas_cache]

            num_nuevos_total = TAMANO_POBLACION - len(elite_tuplas_cache)
            num_inmigrantes = TAMANO_POBLACION // 10
            nuevos_individuos = [crear_individuo() for _ in range(num_inmigrantes)]

            while len(nuevos_individuos) < num_nuevos_total:
                candidatos_padres = random.sample(mejores_genes, min(4, len(mejores_genes)))
                hijo = cruzar_individuos(candidatos_padres[0], candidatos_padres[1])
                nuevos_individuos.append(mutar_individuo(hijo))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    bfit, bs55, bz55, bsp, brip, bgmin, bgmax, baire, blen, campeon = evaluados[0]
    nodos_m = decodificar_alambre_a_cartesianas(campeon)
    nodos_cm = nodos_m * 100.0

    nombre_archivo_nec = "antena_paraguas_55ghz_esp32c5.nec"
    nombre_archivo_scad = "antena_paraguas_55ghz_esp32c5.scad"
    contenido_nec = generar_archivo_nec_campeon(nodos_m, filename=nombre_archivo_nec)
    contenido_scad = generar_archivo_scad_campeon(nodos_cm, filename=nombre_archivo_scad)
    t_total = time.time() - t_inicio

    print("\n" + "=" * 110)
    print(f" ¡ANTENA 5.50 GHz CON PARAGUAS INVERTIDO COMPLETADA EN {t_total:.2f} SEGUNDOS!")
    print(f" * S11 @ 5.50 GHz (Objetivo)     : {bs55:6.2f} dB (Z_in = {bz55.real:5.2f} {bz55.imag:+6.2f}j Ohm)")
    print(f" * Rango Ganancia Domo +Z (0-85°): [{bgmin:+5.2f} dBi a {bgmax:+5.2f} dBi]")
    if PYNEC_DISPONIBLE:
        n55 = simular_un_alambre_pynec(nodos_m, FREQ_55)
        if n55:
            print(f" * PyNEC MoM Hemisferio +Z       : [{n55[3]:+5.2f} dBi a {n55[2]:+5.2f} dBi] (SD={n55[5]:.2f} dB)")
    print(f" * Varianza Global 3D (Spread)   : {bsp:5.2f} dB")
    print(f" * Rizado Azimutal X-Y           : {brip:5.2f} dB")
    print(f" * Longitud brazo evolucionado   : {blen*100:.2f} cm (+ {NUM_RADIALES_PARAGUAS} radiales fijos de {LARGO_RADIAL_M*100:.2f} cm)")
    print(f" * Aire libre auto-intersección  : {baire:.2f} mm")
    print(f" * Archivo NEC2 guardado         : {nombre_archivo_nec}")
    print(f" * Archivo OpenSCAD guardado     : {nombre_archivo_scad}")
    print("=" * 110)

    print(f"\n* --- ARCHIVO NEC2 DEL CAMPEÓN ({nombre_archivo_nec}) ---")
    print(contenido_nec)

    print(f"\n// --- ARCHIVO OPENSCAD DEL CAMPEÓN ({nombre_archivo_scad}) ---")
    print(contenido_scad)

if __name__ == "__main__":
    ejecutar_optimizacion()