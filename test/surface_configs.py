import ettk
import numpy as np

# Page Aruco Size
PAGE_HEIGHT_SIZE = 27
PAGE_WIDTH_SIZE = 21.5
MONITOR_HEIGHT_SIZE = 19
MONITOR_WIDTH_SIZE = 29
M_ARUCO_SIZE = 2.5
W_SCALE = 1/105
H_SCALE = 1/110
# R_CORR = 0.3
R_CORR = 0

# Grid
x_grid = [1.8, 20.0]
y_grid = [4.6, 13.5, 24.0]

# Create configuration for UnwrappingOfThePast
unwrap1_config = ettk.SurfaceConfig(
    id='unwrap1',
    aruco_config={
        8: ettk.ArucoConfig(
            8,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        9: ettk.ArucoConfig(
            9,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        10: ettk.ArucoConfig(
            10,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        11: ettk.ArucoConfig(
            11,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        12: ettk.ArucoConfig(
            12,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        13: ettk.ArucoConfig(
            13,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
unwrap2_config = ettk.SurfaceConfig(
    id='unwrap2',
    aruco_config={
        14: ettk.ArucoConfig(
            14,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        15: ettk.ArucoConfig(
            15,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        16: ettk.ArucoConfig(
            16,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        17: ettk.ArucoConfig(
            17,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        18: ettk.ArucoConfig(
            18,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        19: ettk.ArucoConfig(
            19,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
unwrap3_config = ettk.SurfaceConfig(
    id='unwrap3',
    aruco_config={
        20: ettk.ArucoConfig(
            20,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        21: ettk.ArucoConfig(
            21,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        22: ettk.ArucoConfig(
            22,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        23: ettk.ArucoConfig(
            23,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        24: ettk.ArucoConfig(
            24,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        25: ettk.ArucoConfig(
            25,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)

suffrage1_config = ettk.SurfaceConfig(
    id='suffrage1',
    aruco_config={
        26: ettk.ArucoConfig(
            26,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        27: ettk.ArucoConfig(
            27,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        28: ettk.ArucoConfig(
            28,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        29: ettk.ArucoConfig(
            29,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        30: ettk.ArucoConfig(
            30,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        31: ettk.ArucoConfig(
            31,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
suffrage2_config = ettk.SurfaceConfig(
    id='suffrage2',
    aruco_config={
        32: ettk.ArucoConfig(
            32,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        33: ettk.ArucoConfig(
            33,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        34: ettk.ArucoConfig(
            34,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        35: ettk.ArucoConfig(
            35,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        36: ettk.ArucoConfig(
            36,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        37: ettk.ArucoConfig(
            37,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
suffrage3_config = ettk.SurfaceConfig(
    id='suffrage3',
    aruco_config={
        38: ettk.ArucoConfig(
            38,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        39: ettk.ArucoConfig(
            39,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        40: ettk.ArucoConfig(
            40,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        41: ettk.ArucoConfig(
            41,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        42: ettk.ArucoConfig(
            42,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        43: ettk.ArucoConfig(
            43,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca1_config = ettk.SurfaceConfig(
    id='mooca1',
    aruco_config={
        44: ettk.ArucoConfig(
            44,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        45: ettk.ArucoConfig(
            45,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        46: ettk.ArucoConfig(
            46,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        47: ettk.ArucoConfig(
            47,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        48: ettk.ArucoConfig(
            48,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        49: ettk.ArucoConfig(
            49,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca2_config = ettk.SurfaceConfig(
    id='mooca2',
    aruco_config={
        50: ettk.ArucoConfig(
            50,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        51: ettk.ArucoConfig(
            51,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        52: ettk.ArucoConfig(
            52,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        53: ettk.ArucoConfig(
            53,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        54: ettk.ArucoConfig(
            54,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        55: ettk.ArucoConfig(
            55,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca3_config = ettk.SurfaceConfig(
    id='mooca3',
    aruco_config={
        56: ettk.ArucoConfig(
            56,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        57: ettk.ArucoConfig(
            57,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        58: ettk.ArucoConfig(
            58,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        59: ettk.ArucoConfig(
            59,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        60: ettk.ArucoConfig(
            60,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        61: ettk.ArucoConfig(
            61,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca4_config = ettk.SurfaceConfig(
    id='mooca4',
    aruco_config={
        62: ettk.ArucoConfig(
            62,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        63: ettk.ArucoConfig(
            63,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        64: ettk.ArucoConfig(
            64,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        65: ettk.ArucoConfig(
            65,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        66: ettk.ArucoConfig(
            66,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        67: ettk.ArucoConfig(
            67,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca5_config = ettk.SurfaceConfig(
    id='mooca5',
    aruco_config={
        68: ettk.ArucoConfig(
            68,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        69: ettk.ArucoConfig(
            69,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        70: ettk.ArucoConfig(
            70,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        71: ettk.ArucoConfig(
            71,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        72: ettk.ArucoConfig(
            72,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        73: ettk.ArucoConfig(
            73,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca6_config = ettk.SurfaceConfig(
    id='mooca6',
    aruco_config={
        74: ettk.ArucoConfig(
            74,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        75: ettk.ArucoConfig(
            75,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        76: ettk.ArucoConfig(
            76,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        77: ettk.ArucoConfig(
            77,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        78: ettk.ArucoConfig(
            78,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        79: ettk.ArucoConfig(
            79,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca7_config = ettk.SurfaceConfig(
    id='mooca7',
    aruco_config={
        80: ettk.ArucoConfig(
            80,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        81: ettk.ArucoConfig(
            81,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        82: ettk.ArucoConfig(
            82,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        83: ettk.ArucoConfig(
            83,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        84: ettk.ArucoConfig(
            84,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        85: ettk.ArucoConfig(
            85,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca8_config = ettk.SurfaceConfig(
    id='mooca8',
    aruco_config={
        86: ettk.ArucoConfig(
            86,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        87: ettk.ArucoConfig(
            87,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        88: ettk.ArucoConfig(
            88,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        89: ettk.ArucoConfig(
            89,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        90: ettk.ArucoConfig(
            90,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        91: ettk.ArucoConfig(
            91,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca9_config = ettk.SurfaceConfig(
    id='mooca9',
    aruco_config={
        92: ettk.ArucoConfig(
            92,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        93: ettk.ArucoConfig(
            93,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        94: ettk.ArucoConfig(
            94,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        95: ettk.ArucoConfig(
            95,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        96: ettk.ArucoConfig(
            96,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        97: ettk.ArucoConfig(
            97,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
mooca10_config = ettk.SurfaceConfig(
    id='mooca10',
    aruco_config={
        98: ettk.ArucoConfig(
            98,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[0], 0])
        ),
        99: ettk.ArucoConfig(
            99,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        100: ettk.ArucoConfig(
            100,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        101: ettk.ArucoConfig(
            101,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        102: ettk.ArucoConfig(
            102,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        103: ettk.ArucoConfig(
            103,
            np.array([R_CORR, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
    },
    height=PAGE_HEIGHT_SIZE,
    width=PAGE_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)

monitor_config = ettk.SurfaceConfig(
    id='monitor',
    aruco_config={
        0: ettk.ArucoConfig(
            0,
            np.array([0, 0, 0]),
            np.array([0, 0, 0])
        ),
        1: ettk.ArucoConfig(
            1,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[0], 0])
        ),
        2: ettk.ArucoConfig(
            2,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[1], 0])
        ),
        3: ettk.ArucoConfig(
            3,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[1], 0])
        ),
        4: ettk.ArucoConfig(
            4,
            np.array([0, 0, 0]),
            np.array([x_grid[0], y_grid[2], 0])
        ),
        5: ettk.ArucoConfig(
            5,
            np.array([0, 0, 0]),
            np.array([x_grid[1], y_grid[2], 0])
        ),
        6: ettk.ArucoConfig(
            6,
            # np.array([1, -np.pi*2+1, 0]),
            np.array([1, -5.14, 0]),
            np.array([-M_ARUCO_SIZE/2, 1.2+M_ARUCO_SIZE/2, 0])
        ),
        7: ettk.ArucoConfig(
            7,
            np.array([0, -0.03, -0.05]),
            np.array([2.3, -M_ARUCO_SIZE/2, 0])
        ),
    },
    height=MONITOR_HEIGHT_SIZE,
    width=MONITOR_WIDTH_SIZE,
    scale=(W_SCALE, H_SCALE)
)
