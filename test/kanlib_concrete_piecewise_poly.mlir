module {
  func.func private @print_f32(f32)

  func.func @kanlib_piecewise_concrete()
      -> tensor<8x8xf32> {

    %x = arith.constant dense<[[-0.899999976, -0.75], [-0.550000012, -0.349999994], [-0.100000001, 0.100000001], [0.349999994, 0.5], [0.75, 0.899999976], [-0.800000012, 0.0], [0.0, 0.800000012], [0.589999974, -0.589999974]]>
      : tensor<8x2xf32>

    %bounds = arith.constant dense<[[-1.0, -0.600000024, -0.199999988, 0.200000048, 0.600000024, 1.0], [-1.0, -0.600000024, -0.199999988, 0.200000048, 0.600000024, 1.0]]>
      : tensor<2x6xf32>

    %coeffs = arith.constant dense<[[[[-0.224581093, -0.883360088, -1.28286219, -0.524318933], [0.0321639664, 0.400365114, 0.856679618, 0.664315403], [0.0207969882, 0.22986041, 0.00415613968, -0.756556988], [0.0135195991, 0.339021295, -0.541648388, 0.153117031], [0.043210119, 0.190568566, -0.294226974, 0.0156606138]], [[-0.333976924, -1.78946888, -2.79259753, -1.26408756], [0.0239317697, 7.45256693e-05, 0.189974621, 0.392896831], [0.0238147415, -0.00168086798, 0.181197733, 0.378268719], [0.0321425833, -0.126598388, 0.80578512, -0.662709951], [-0.110750817, 0.587868631, -0.384993374, -0.00116631982]]], [[[-0.280663848, -1.7576623, -2.81190658, -1.20874584], [0.0712239519, 0.00177649444, 0.120491333, 0.420363903], [0.0646574125, -0.0967215523, -0.371998727, -0.400453001], [0.0503559932, 0.11779964, -1.4446044, 1.38722277], [0.484267324, -2.05175662, 2.17132187, -0.621624887]], [[-0.0682194382, -0.431685627, -0.862421095, -0.469534934], [0.0840635896, 0.329729527, 0.406604141, 0.235479057], [0.0783090666, 0.24341163, -0.0249853916, -0.483836919], [0.0740802214, 0.306844354, -0.342149287, 0.0447697677], [0.101353027, 0.170480192, -0.114875503, -0.0814935192]]], [[[0.267394394, 1.18640292, 1.53674138, 0.634064376], [0.0283330455, -0.00890381448, -0.455436349, -0.472700953], [0.0321797431, 0.0487966761, -0.16693379, 0.00813670922], [0.0293818545, 0.0907650217, -0.376775503, 0.357872814], [0.195764586, -0.741148651, 1.00974727, -0.412417561]], [[0.290823251, 1.39818978, 2.03938007, 0.885716856], [0.026602244, 0.0770848468, -0.16246143, -0.337528348], [0.0317624994, 0.154488668, 0.224557742, 0.30750376], [0.0485130437, -0.0967693627, 1.48084736, -1.78631175], [-0.855041623, 4.42100334, -6.04877281, 2.39681029]]], [[[-0.0664466098, -0.399070472, -0.189065605, 0.0789983943], [0.00251716818, -0.0542518944, 0.385631621, 0.398274451], [-0.00113272143, -0.109000243, 0.111889817, -0.0579620861], [-0.00336664473, -0.075491406, -0.0556543134, 0.221278086], [0.211490318, -1.1497761, 1.73481989, -0.773429692]], [[0.581618786, 1.17966461, 1.06291461, 0.397354722], [0.0489530452, -1.48366451, -3.37596774, -2.06869125], [0.0906482786, -0.858235836, -0.248824134, 3.14321423], [0.120964713, -1.3129822, 2.0249064, -0.646335125], [0.44159019, -2.91610885, 4.69678307, -2.1307106]]], [[[0.209672406, 0.403670609, 0.599404752, 0.335048944], [-0.0278098341, -0.783740699, -1.37961423, -0.764406085], [-0.00136009254, -0.386994511, 0.604116797, 2.54181242], [0.0479311496, -1.12636292, 4.30095768, -3.61958742], [-1.34898174, 5.85820103, -7.33998108, 2.84760046]], [[-0.00407708576, 0.231325284, 0.403161526, 0.239561409], [-0.0815676674, -0.156127632, -0.242593303, -0.119191252], [-0.0737460777, -0.0388037637, 0.344026029, 0.858507812], [-0.0539889336, -0.335160822, 1.82581091, -1.61113298], [-0.719219446, 2.99099135, -3.71777606, 1.46863735]]], [[[0.432837158, 2.11447787, 2.92815018, 1.21623385], [0.0372243226, 0.136413828, -0.368622869, -0.615306556], [0.037803594, 0.145102903, -0.325177372, -0.542897224], [0.023534663, 0.35913676, -1.3953464, 1.24071753], [0.561655223, -2.33146596, 3.08899093, -1.25058091]], [[1.12962663, 5.5476594, 7.85609722, 3.32002449], [-0.0849390402, -0.525168419, -2.26528215, -2.3029635], [-0.0583675392, -0.126595899, -0.272419482, 1.0184747], [-0.0498402044, -0.254505903, 0.36713028, -0.0474413708], [0.00825819001, -0.544997692, 0.851282954, -0.316414922]]], [[[-0.373103738, -2.09358954, -3.40304255, -1.62505281], [-0.12216448, -0.838893056, -1.3118819, -0.46329689], [-0.103127837, -0.553343415, 0.1158664, 1.91628349], [-0.0805086866, -0.892630517, 1.81230164, -0.911107719], [-0.192290366, -0.333721846, 0.880786777, -0.393599331]], [[0.0110274954, -0.169048324, -0.315170199, -0.149969041], [0.0451569967, 0.00159916386, -0.0307577327, 0.00803787168], [0.04256979, -0.0372089073, -0.224797994, -0.315362632], [0.0311013777, 0.134817228, -1.08492839, 1.11818779], [0.628342032, -2.85138583, 3.89207649, -1.6468147]]], [[[-0.515508831, -2.51215029, -3.96113491, -1.79729319], [0.0572514907, 0.351651132, 0.811867356, 0.854374647], [0.0450676791, 0.168893948, -0.101918444, -0.668601751], [0.037734326, 0.278894186, -0.651919484, 0.248066396], [-0.0657997578, 0.7965644, -1.51470292, 0.727390349]], [[-0.801757395, -3.70939231, -4.72324276, -1.80634427], [0.013697328, 0.367880791, 2.07221174, 1.96890783], [-0.00692130206, 0.0586012565, 0.525813878, -0.608422518], [-0.00804549549, 0.0754642338, 0.441498756, -0.467897087], [-0.22091502, 1.13981175, -1.33241379, 0.517609954]]]]>
      : tensor<8x2x5x4xf32>

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<8x2xf32>,
          tensor<2x6xf32>,
          tensor<8x2x5x4xf32>
          -> tensor<8x8xf32>

    return %y : tensor<8x8xf32>
  }

  func.func @main() {
    %result = func.call @kanlib_piecewise_concrete()
      : () -> tensor<8x8xf32>

    %c_b0_o0_b = arith.constant 0 : index
    %c_b0_o0_o = arith.constant 0 : index
    %v_b0_o0 = tensor.extract %result[
        %c_b0_o0_b,
        %c_b0_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o0) : (f32) -> ()

    %c_b0_o1_b = arith.constant 0 : index
    %c_b0_o1_o = arith.constant 1 : index
    %v_b0_o1 = tensor.extract %result[
        %c_b0_o1_b,
        %c_b0_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o1) : (f32) -> ()

    %c_b0_o2_b = arith.constant 0 : index
    %c_b0_o2_o = arith.constant 2 : index
    %v_b0_o2 = tensor.extract %result[
        %c_b0_o2_b,
        %c_b0_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o2) : (f32) -> ()

    %c_b0_o3_b = arith.constant 0 : index
    %c_b0_o3_o = arith.constant 3 : index
    %v_b0_o3 = tensor.extract %result[
        %c_b0_o3_b,
        %c_b0_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o3) : (f32) -> ()

    %c_b0_o4_b = arith.constant 0 : index
    %c_b0_o4_o = arith.constant 4 : index
    %v_b0_o4 = tensor.extract %result[
        %c_b0_o4_b,
        %c_b0_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o4) : (f32) -> ()

    %c_b0_o5_b = arith.constant 0 : index
    %c_b0_o5_o = arith.constant 5 : index
    %v_b0_o5 = tensor.extract %result[
        %c_b0_o5_b,
        %c_b0_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o5) : (f32) -> ()

    %c_b0_o6_b = arith.constant 0 : index
    %c_b0_o6_o = arith.constant 6 : index
    %v_b0_o6 = tensor.extract %result[
        %c_b0_o6_b,
        %c_b0_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o6) : (f32) -> ()

    %c_b0_o7_b = arith.constant 0 : index
    %c_b0_o7_o = arith.constant 7 : index
    %v_b0_o7 = tensor.extract %result[
        %c_b0_o7_b,
        %c_b0_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b0_o7) : (f32) -> ()

    %c_b1_o0_b = arith.constant 1 : index
    %c_b1_o0_o = arith.constant 0 : index
    %v_b1_o0 = tensor.extract %result[
        %c_b1_o0_b,
        %c_b1_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o0) : (f32) -> ()

    %c_b1_o1_b = arith.constant 1 : index
    %c_b1_o1_o = arith.constant 1 : index
    %v_b1_o1 = tensor.extract %result[
        %c_b1_o1_b,
        %c_b1_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o1) : (f32) -> ()

    %c_b1_o2_b = arith.constant 1 : index
    %c_b1_o2_o = arith.constant 2 : index
    %v_b1_o2 = tensor.extract %result[
        %c_b1_o2_b,
        %c_b1_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o2) : (f32) -> ()

    %c_b1_o3_b = arith.constant 1 : index
    %c_b1_o3_o = arith.constant 3 : index
    %v_b1_o3 = tensor.extract %result[
        %c_b1_o3_b,
        %c_b1_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o3) : (f32) -> ()

    %c_b1_o4_b = arith.constant 1 : index
    %c_b1_o4_o = arith.constant 4 : index
    %v_b1_o4 = tensor.extract %result[
        %c_b1_o4_b,
        %c_b1_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o4) : (f32) -> ()

    %c_b1_o5_b = arith.constant 1 : index
    %c_b1_o5_o = arith.constant 5 : index
    %v_b1_o5 = tensor.extract %result[
        %c_b1_o5_b,
        %c_b1_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o5) : (f32) -> ()

    %c_b1_o6_b = arith.constant 1 : index
    %c_b1_o6_o = arith.constant 6 : index
    %v_b1_o6 = tensor.extract %result[
        %c_b1_o6_b,
        %c_b1_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o6) : (f32) -> ()

    %c_b1_o7_b = arith.constant 1 : index
    %c_b1_o7_o = arith.constant 7 : index
    %v_b1_o7 = tensor.extract %result[
        %c_b1_o7_b,
        %c_b1_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b1_o7) : (f32) -> ()

    %c_b2_o0_b = arith.constant 2 : index
    %c_b2_o0_o = arith.constant 0 : index
    %v_b2_o0 = tensor.extract %result[
        %c_b2_o0_b,
        %c_b2_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o0) : (f32) -> ()

    %c_b2_o1_b = arith.constant 2 : index
    %c_b2_o1_o = arith.constant 1 : index
    %v_b2_o1 = tensor.extract %result[
        %c_b2_o1_b,
        %c_b2_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o1) : (f32) -> ()

    %c_b2_o2_b = arith.constant 2 : index
    %c_b2_o2_o = arith.constant 2 : index
    %v_b2_o2 = tensor.extract %result[
        %c_b2_o2_b,
        %c_b2_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o2) : (f32) -> ()

    %c_b2_o3_b = arith.constant 2 : index
    %c_b2_o3_o = arith.constant 3 : index
    %v_b2_o3 = tensor.extract %result[
        %c_b2_o3_b,
        %c_b2_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o3) : (f32) -> ()

    %c_b2_o4_b = arith.constant 2 : index
    %c_b2_o4_o = arith.constant 4 : index
    %v_b2_o4 = tensor.extract %result[
        %c_b2_o4_b,
        %c_b2_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o4) : (f32) -> ()

    %c_b2_o5_b = arith.constant 2 : index
    %c_b2_o5_o = arith.constant 5 : index
    %v_b2_o5 = tensor.extract %result[
        %c_b2_o5_b,
        %c_b2_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o5) : (f32) -> ()

    %c_b2_o6_b = arith.constant 2 : index
    %c_b2_o6_o = arith.constant 6 : index
    %v_b2_o6 = tensor.extract %result[
        %c_b2_o6_b,
        %c_b2_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o6) : (f32) -> ()

    %c_b2_o7_b = arith.constant 2 : index
    %c_b2_o7_o = arith.constant 7 : index
    %v_b2_o7 = tensor.extract %result[
        %c_b2_o7_b,
        %c_b2_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b2_o7) : (f32) -> ()

    %c_b3_o0_b = arith.constant 3 : index
    %c_b3_o0_o = arith.constant 0 : index
    %v_b3_o0 = tensor.extract %result[
        %c_b3_o0_b,
        %c_b3_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o0) : (f32) -> ()

    %c_b3_o1_b = arith.constant 3 : index
    %c_b3_o1_o = arith.constant 1 : index
    %v_b3_o1 = tensor.extract %result[
        %c_b3_o1_b,
        %c_b3_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o1) : (f32) -> ()

    %c_b3_o2_b = arith.constant 3 : index
    %c_b3_o2_o = arith.constant 2 : index
    %v_b3_o2 = tensor.extract %result[
        %c_b3_o2_b,
        %c_b3_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o2) : (f32) -> ()

    %c_b3_o3_b = arith.constant 3 : index
    %c_b3_o3_o = arith.constant 3 : index
    %v_b3_o3 = tensor.extract %result[
        %c_b3_o3_b,
        %c_b3_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o3) : (f32) -> ()

    %c_b3_o4_b = arith.constant 3 : index
    %c_b3_o4_o = arith.constant 4 : index
    %v_b3_o4 = tensor.extract %result[
        %c_b3_o4_b,
        %c_b3_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o4) : (f32) -> ()

    %c_b3_o5_b = arith.constant 3 : index
    %c_b3_o5_o = arith.constant 5 : index
    %v_b3_o5 = tensor.extract %result[
        %c_b3_o5_b,
        %c_b3_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o5) : (f32) -> ()

    %c_b3_o6_b = arith.constant 3 : index
    %c_b3_o6_o = arith.constant 6 : index
    %v_b3_o6 = tensor.extract %result[
        %c_b3_o6_b,
        %c_b3_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o6) : (f32) -> ()

    %c_b3_o7_b = arith.constant 3 : index
    %c_b3_o7_o = arith.constant 7 : index
    %v_b3_o7 = tensor.extract %result[
        %c_b3_o7_b,
        %c_b3_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b3_o7) : (f32) -> ()

    %c_b4_o0_b = arith.constant 4 : index
    %c_b4_o0_o = arith.constant 0 : index
    %v_b4_o0 = tensor.extract %result[
        %c_b4_o0_b,
        %c_b4_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o0) : (f32) -> ()

    %c_b4_o1_b = arith.constant 4 : index
    %c_b4_o1_o = arith.constant 1 : index
    %v_b4_o1 = tensor.extract %result[
        %c_b4_o1_b,
        %c_b4_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o1) : (f32) -> ()

    %c_b4_o2_b = arith.constant 4 : index
    %c_b4_o2_o = arith.constant 2 : index
    %v_b4_o2 = tensor.extract %result[
        %c_b4_o2_b,
        %c_b4_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o2) : (f32) -> ()

    %c_b4_o3_b = arith.constant 4 : index
    %c_b4_o3_o = arith.constant 3 : index
    %v_b4_o3 = tensor.extract %result[
        %c_b4_o3_b,
        %c_b4_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o3) : (f32) -> ()

    %c_b4_o4_b = arith.constant 4 : index
    %c_b4_o4_o = arith.constant 4 : index
    %v_b4_o4 = tensor.extract %result[
        %c_b4_o4_b,
        %c_b4_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o4) : (f32) -> ()

    %c_b4_o5_b = arith.constant 4 : index
    %c_b4_o5_o = arith.constant 5 : index
    %v_b4_o5 = tensor.extract %result[
        %c_b4_o5_b,
        %c_b4_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o5) : (f32) -> ()

    %c_b4_o6_b = arith.constant 4 : index
    %c_b4_o6_o = arith.constant 6 : index
    %v_b4_o6 = tensor.extract %result[
        %c_b4_o6_b,
        %c_b4_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o6) : (f32) -> ()

    %c_b4_o7_b = arith.constant 4 : index
    %c_b4_o7_o = arith.constant 7 : index
    %v_b4_o7 = tensor.extract %result[
        %c_b4_o7_b,
        %c_b4_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b4_o7) : (f32) -> ()

    %c_b5_o0_b = arith.constant 5 : index
    %c_b5_o0_o = arith.constant 0 : index
    %v_b5_o0 = tensor.extract %result[
        %c_b5_o0_b,
        %c_b5_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o0) : (f32) -> ()

    %c_b5_o1_b = arith.constant 5 : index
    %c_b5_o1_o = arith.constant 1 : index
    %v_b5_o1 = tensor.extract %result[
        %c_b5_o1_b,
        %c_b5_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o1) : (f32) -> ()

    %c_b5_o2_b = arith.constant 5 : index
    %c_b5_o2_o = arith.constant 2 : index
    %v_b5_o2 = tensor.extract %result[
        %c_b5_o2_b,
        %c_b5_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o2) : (f32) -> ()

    %c_b5_o3_b = arith.constant 5 : index
    %c_b5_o3_o = arith.constant 3 : index
    %v_b5_o3 = tensor.extract %result[
        %c_b5_o3_b,
        %c_b5_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o3) : (f32) -> ()

    %c_b5_o4_b = arith.constant 5 : index
    %c_b5_o4_o = arith.constant 4 : index
    %v_b5_o4 = tensor.extract %result[
        %c_b5_o4_b,
        %c_b5_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o4) : (f32) -> ()

    %c_b5_o5_b = arith.constant 5 : index
    %c_b5_o5_o = arith.constant 5 : index
    %v_b5_o5 = tensor.extract %result[
        %c_b5_o5_b,
        %c_b5_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o5) : (f32) -> ()

    %c_b5_o6_b = arith.constant 5 : index
    %c_b5_o6_o = arith.constant 6 : index
    %v_b5_o6 = tensor.extract %result[
        %c_b5_o6_b,
        %c_b5_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o6) : (f32) -> ()

    %c_b5_o7_b = arith.constant 5 : index
    %c_b5_o7_o = arith.constant 7 : index
    %v_b5_o7 = tensor.extract %result[
        %c_b5_o7_b,
        %c_b5_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b5_o7) : (f32) -> ()

    %c_b6_o0_b = arith.constant 6 : index
    %c_b6_o0_o = arith.constant 0 : index
    %v_b6_o0 = tensor.extract %result[
        %c_b6_o0_b,
        %c_b6_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o0) : (f32) -> ()

    %c_b6_o1_b = arith.constant 6 : index
    %c_b6_o1_o = arith.constant 1 : index
    %v_b6_o1 = tensor.extract %result[
        %c_b6_o1_b,
        %c_b6_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o1) : (f32) -> ()

    %c_b6_o2_b = arith.constant 6 : index
    %c_b6_o2_o = arith.constant 2 : index
    %v_b6_o2 = tensor.extract %result[
        %c_b6_o2_b,
        %c_b6_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o2) : (f32) -> ()

    %c_b6_o3_b = arith.constant 6 : index
    %c_b6_o3_o = arith.constant 3 : index
    %v_b6_o3 = tensor.extract %result[
        %c_b6_o3_b,
        %c_b6_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o3) : (f32) -> ()

    %c_b6_o4_b = arith.constant 6 : index
    %c_b6_o4_o = arith.constant 4 : index
    %v_b6_o4 = tensor.extract %result[
        %c_b6_o4_b,
        %c_b6_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o4) : (f32) -> ()

    %c_b6_o5_b = arith.constant 6 : index
    %c_b6_o5_o = arith.constant 5 : index
    %v_b6_o5 = tensor.extract %result[
        %c_b6_o5_b,
        %c_b6_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o5) : (f32) -> ()

    %c_b6_o6_b = arith.constant 6 : index
    %c_b6_o6_o = arith.constant 6 : index
    %v_b6_o6 = tensor.extract %result[
        %c_b6_o6_b,
        %c_b6_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o6) : (f32) -> ()

    %c_b6_o7_b = arith.constant 6 : index
    %c_b6_o7_o = arith.constant 7 : index
    %v_b6_o7 = tensor.extract %result[
        %c_b6_o7_b,
        %c_b6_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b6_o7) : (f32) -> ()

    %c_b7_o0_b = arith.constant 7 : index
    %c_b7_o0_o = arith.constant 0 : index
    %v_b7_o0 = tensor.extract %result[
        %c_b7_o0_b,
        %c_b7_o0_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o0) : (f32) -> ()

    %c_b7_o1_b = arith.constant 7 : index
    %c_b7_o1_o = arith.constant 1 : index
    %v_b7_o1 = tensor.extract %result[
        %c_b7_o1_b,
        %c_b7_o1_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o1) : (f32) -> ()

    %c_b7_o2_b = arith.constant 7 : index
    %c_b7_o2_o = arith.constant 2 : index
    %v_b7_o2 = tensor.extract %result[
        %c_b7_o2_b,
        %c_b7_o2_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o2) : (f32) -> ()

    %c_b7_o3_b = arith.constant 7 : index
    %c_b7_o3_o = arith.constant 3 : index
    %v_b7_o3 = tensor.extract %result[
        %c_b7_o3_b,
        %c_b7_o3_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o3) : (f32) -> ()

    %c_b7_o4_b = arith.constant 7 : index
    %c_b7_o4_o = arith.constant 4 : index
    %v_b7_o4 = tensor.extract %result[
        %c_b7_o4_b,
        %c_b7_o4_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o4) : (f32) -> ()

    %c_b7_o5_b = arith.constant 7 : index
    %c_b7_o5_o = arith.constant 5 : index
    %v_b7_o5 = tensor.extract %result[
        %c_b7_o5_b,
        %c_b7_o5_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o5) : (f32) -> ()

    %c_b7_o6_b = arith.constant 7 : index
    %c_b7_o6_o = arith.constant 6 : index
    %v_b7_o6 = tensor.extract %result[
        %c_b7_o6_b,
        %c_b7_o6_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o6) : (f32) -> ()

    %c_b7_o7_b = arith.constant 7 : index
    %c_b7_o7_o = arith.constant 7 : index
    %v_b7_o7 = tensor.extract %result[
        %c_b7_o7_b,
        %c_b7_o7_o
    ] : tensor<8x8xf32>
    func.call @print_f32(%v_b7_o7) : (f32) -> ()

    return
  }
}
