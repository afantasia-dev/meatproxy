// =====================================================================
// MODELO 3D OPENSCAD: ANTENA EN CRUZ DE DIPOLOS (ESP32-C5: 2.44 & 5.50 GHz)
// Unidades base de OpenSCAD: Milímetros (mm)
// Coordenadas de entrada: Centímetros (cm) -> convertidas automáticamente
// =====================================================================

$fn = 48; // Resolución de las curvas tubulares (subir a 64 para exportar STL final)

// --- PARÁMETROS FÍSICOS Y DELIMITADORES ---
radio_alambre_mm = 2;       // Alambre de cobre de 1.0 mm de diámetro (0.5 mm radio)
radio_cilindro_cm = 4.0;      // 4.0 cm de radio máximo del cilindro único
altura_max_24_cm = 7.0;       // 7.0 cm de altura máxima para la rama de 2.44 GHz
altura_max_55_cm = 3.5;       // 3.5 cm de altura máxima para la rama de 5.50 GHz

mostrar_envolvente = true;    // Cambiar a false si deseas exportar solo el alambre a STL
mostrar_ejes_guia = true;     // Muestra la cruz ortogonal de referencia en la base

// =====================================================================
// 1. COORDENADAS OPTIMIZADAS (EN CENTÍMETROS)
// Reemplaza estas listas si corres el script con otra semilla o parámetros
// =====================================================================
nodos_24 = [
    [  0.000,   0.000,   0.000], // Paso 0
    [  2.391,   0.053,   0.709], // Paso 1
    [  2.468,  -2.726,   1.221], // Paso 2
    [  2.178,  -2.809,   2.293], // Paso 3
    [ -0.699,  -1.026,   3.515], // Paso 4
    [ -1.054,  -0.689,   3.933], // Paso 5
    [ -3.354,  -0.477,   4.742], // Paso 6
    [ -2.786,   0.838,   6.022], // Paso 7
    [ -1.215,   1.487,   7.000] // Paso 8
];

nodos_55 = [
    [  0.000,   0.000,   0.000], // Paso 0
    [ -0.133,   1.060,   0.466], // Paso 1
    [ -0.707,   2.819,   0.821], // Paso 2
    [ -2.388,   2.139,   1.103], // Paso 3
    [ -2.487,   0.636,   1.253], // Paso 4
    [ -3.041,  -0.120,   1.403], // Paso 5
    [ -2.197,  -0.442,   2.103], // Paso 6
    [ -1.535,  -0.049,   2.420], // Paso 7
    [ -3.118,   1.535,   2.594] // Paso 8
];







// =====================================================================
// 2. MÓDULOS DE CONSTRUCCIÓN Y SUAVIZADO (HULL + SPHERE)
// =====================================================================

// Convierte una lista de nodos en cm a una estructura tubular continua en mm
module trazar_rama_alambre(nodos_cm, radio_mm) {
    for (i = [0 : len(nodos_cm) - 2]) {
        p1_mm = nodos_cm[i] * 10;     // Conversión de cm a mm
        p2_mm = nodos_cm[i + 1] * 10; // Conversión de cm a mm
        
        hull() {
            translate(p1_mm) sphere(r = radio_mm);
            translate(p2_mm) sphere(r = radio_mm);
        }
    }
}

// Punto de alimentación común y plano de referencia del ESP32-C5
module pin_alimentacion_esp32() {
    // Pin de soldadura RF en (0,0,0)
    color([0.85, 0.85, 0.90, 1.0])
    translate([0, 0, -3])
        cylinder(h = 4, r = 1.2);
    
    // Pequeño disco indicador del nodo común en (0,0,0)
    color([0.2, 0.2, 0.2, 0.9])
    translate([0, 0, -1])
        cylinder(h = 1, r = 4);

    if (mostrar_ejes_guia) {
        // Guía Eje X (Dirección inicial Rama 2.44 GHz)
        color([1.0, 0.8, 0.0, 0.4])
        translate([0, -0.25, -0.2])
            cube([radio_cilindro_cm * 10, 0.5, 0.2]);
            
        // Guía Eje Y (Dirección inicial ortogonal 90° Rama 5.50 GHz)
        color([1.0, 0.3, 0.0, 0.4])
        translate([-0.25, 0, -0.2])
            cube([0.5, radio_cilindro_cm * 10, 0.2]);
    }
}

// Envolventes cilíndricas transparentes para verificar límites espaciales
module envolvente_cilindrica() {
    // Cilindro delimitador total (H = 7.0 cm para 2.44 GHz, R = 4.0 cm)
    color([0.6, 0.8, 1.0, 0.12])
    cylinder(h = altura_max_24_cm * 10, r = radio_cilindro_cm * 10);
    
    // Anillo marcador a la mitad de altura (H = 3.5 cm, techo de la rama de 5.50 GHz)
    color([1.0, 0.4, 0.1, 0.25])
    translate([0, 0, altura_max_55_cm * 10 - 0.2])
        difference() {
            cylinder(h = 0.4, r = radio_cilindro_cm * 10 + 0.3);
            translate([0, 0, -0.1])
                cylinder(h = 0.6, r = radio_cilindro_cm * 10 - 0.5);
        }
}

// =====================================================================
// 3. RENDERIZADO PRINCIPAL DE LA ESCENA
// =====================================================================

// 1. Renderizar Rama de 2.44 GHz (Color Dorado Brillante)
color([1.0, 0.78, 0.10, 1.0])
trazar_rama_alambre(nodos_24, radio_alambre_mm);

// 2. Renderizar Rama de 5.50 GHz (Color Cobre / Anaranjado Rojizo)
color([0.95, 0.35, 0.10, 1.0])
trazar_rama_alambre(nodos_55, radio_alambre_mm);

// 3. Renderizar Base de Alimentación (Pin ANT del ESP32-C5)
pin_alimentacion_esp32();

// 4. Renderizar Cilindro Delimitador Transparente (al final para correcta transparencia en OpenSCAD)
if (mostrar_envolvente) {
    envolvente_cilindrica();
}