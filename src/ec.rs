// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::ec::*;
  } else {
    pub use self::rust::*;
  }
}

use crate::context::CDFContextLog;
use crate::util::{msb, ILog};
use bitstream_io::{BigEndian, BitWrite, BitWriter};
use std::io;

pub const OD_BITRES: u8 = 3;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
type ec_window = u32;

/// Public trait interface to a bitstream `Writer`: a `Counter` can be
/// used to count bits for cost analysis without actually storing
/// anything (using a new `WriterCounter` as a `Writer`), to record
/// tokens for later writing (using a new `WriterRecorder` as a
/// `Writer`) to write actual final bits out using a range encoder
/// (using a new `WriterEncoder` as a `Writer`).  A `WriterRecorder`'s
/// contents can be replayed into a `WriterEncoder`.
pub trait Writer {
  /// Write a symbol `s`, using the passed in cdf reference; leaves `cdf` unchanged
  fn symbol<const CDF_LEN: usize>(&mut self, s: u32, cdf: &[u16; CDF_LEN]);
  /// return approximate number of fractional bits in `OD_BITRES`
  /// precision to write a symbol `s` using the passed in cdf reference;
  /// leaves `cdf` unchanged
  fn symbol_bits(&self, s: u32, cdf: &[u16]) -> u32;
  /// Write a symbol `s`, using the passed in cdf reference; updates the referenced cdf.
  fn symbol_with_update<const CDF_LEN: usize>(
    &mut self, s: u32, cdf: &mut [u16; CDF_LEN], log: &mut CDFContextLog,
  );
  /// Write a bool using passed in probability
  fn bool(&mut self, val: bool, f: u16);
  /// Write a single bit with flat probability
  fn bit(&mut self, bit: u16);
  /// Write literal `bits` with flat probability
  fn literal(&mut self, bits: u8, s: u32);
  /// Write passed `level` as a golomb code
  fn write_golomb(&mut self, level: u32);
  /// Write a value `v` in `[0, n-1]` quasi-uniformly
  fn write_quniform(&mut self, n: u32, v: u32);
  /// Return fractional bits needed to write a value `v` in `[0, n-1]`
  /// quasi-uniformly
  fn count_quniform(&self, n: u32, v: u32) -> u32;
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite subexponential
  fn write_subexp(&mut self, n: u32, k: u8, v: u32);
  /// Return fractional bits needed to write symbol v in `[0, n-1]` with
  /// parameter k as finite subexponential
  fn count_subexp(&self, n: u32, k: u8, v: u32) -> u32;
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[0, n-1]`.
  fn write_unsigned_subexp_with_ref(&mut self, v: u32, mx: u32, k: u8, r: u32);
  /// Return fractional bits needed to write symbol `v` in `[0, n-1]` with
  /// parameter `k` as finite subexponential based on a reference `r`
  /// also in `[0, n-1]`.
  fn count_unsigned_subexp_with_ref(
    &self, v: u32, mx: u32, k: u8, r: u32,
  ) -> u32;
  /// Write symbol v in `[-(n-1), n-1]` with parameter k as finite
  /// subexponential based on a reference ref also in `[-(n-1), n-1]`.
  fn write_signed_subexp_with_ref(
    &mut self, v: i32, low: i32, high: i32, k: u8, r: i32,
  );
  /// Return fractional bits needed to write symbol `v` in `[-(n-1), n-1]`
  /// with parameter `k` as finite subexponential based on a reference
  /// `r` also in `[-(n-1), n-1]`.
  fn count_signed_subexp_with_ref(
    &self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) -> u32;
  /// Return current length of range-coded bitstream in integer bits
  fn tell(&mut self) -> u32;
  /// Return current length of range-coded bitstream in fractional
  /// bits with `OD_BITRES` decimal precision
  fn tell_frac(&mut self) -> u32;
  /// Save current point in coding/recording to a checkpoint
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Restore saved position in coding/recording from a checkpoint
  fn rollback(&mut self, _: &WriterCheckpoint);
  /// Add additional bits from rate estimators without coding a real symbol
  fn add_bits_frac(&mut self, bits_frac: u32);
}

/// `StorageBackend` is an internal trait used to tie a specific `Writer`
/// implementation's storage to the generic `Writer`.  It would be
/// private, but Rust is deprecating 'private trait in a public
/// interface' support.
pub trait StorageBackend {
  /// Store partially-computed range code into given storage backend
  fn store(&mut self, fl: u16, fh: u16, nms: u16);
  /// Return byte-length of encoded stream to date
  fn stream_bits(&mut self) -> usize;
  /// Backend implementation of checkpoint to pass through Writer interface
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Backend implementation of rollback to pass through Writer interface
  fn rollback(&mut self, _: &WriterCheckpoint);
}

#[derive(Debug, Clone)]
pub struct WriterBase<S> {
  /// The number of values in the current range.
  rng: u16,
  /// The number of bits of data in the current value.
  cnt: i16,
  #[cfg(feature = "desync_finder")]
  /// Debug enable flag
  debug: bool,
  /// Extra offset added to tell() and tell_frac() to approximate costs
  /// of actually coding a symbol
  fake_bits_frac: u32,
  /// Use-specific storage
  s: S,
}

//#[allow(clippy:)]
/*
 * import math
 * def lr_compute(rngShift, diff):
 *     u = ((rngShift * diff) >> 1) + 4
 *     bits = 15 - math.floor(math.log2(u))
 *     u = u << bits
 *     return ((u >> 8) - 128, bits)
 * 
 * def stats():
 *     for diff in range(0, 513):
 *         avg = sum([lr_compute(r, diff)[1] for r in range(128, 256)])
 *         yield(avg)
 * 
 * print([x for x in (stats())])
 */
// Units of 1/128 of a bit
const ENTROPY_LOOKUP: [u16; 513] = [1664, 1144, 1020, 936, 894, 844, 810, 786, 767, 739, 716, 698, 682, 669, 658, 648, 640, 625, 612, 600, 589, 579, 570, 562, 555, 548, 542, 536, 530, 525, 521, 516, 512, 504, 497, 490, 484, 478, 472, 466, 461, 456, 451, 447, 442, 438, 434, 431, 427, 424, 420, 417, 414, 411, 408, 405, 403, 400, 398, 395, 393, 391, 388, 386, 384, 380, 377, 373, 369, 366, 362, 359, 356, 353, 350, 347, 344, 341, 338, 336, 333, 331, 328, 326, 323, 321, 319, 317, 315, 312, 310, 308, 306, 305, 303, 301, 299, 297, 296, 294, 292, 291, 289, 287, 286, 284, 283, 282, 280, 279, 277, 276, 275, 273, 272, 271, 270, 268, 267, 266, 265, 264, 263, 262, 261, 260, 258, 257, 
256, 254, 252, 251, 249, 247, 245, 243, 241, 240, 238, 236, 234, 233, 231, 230, 228, 226, 225, 223, 222, 220, 219, 217, 216, 215, 213, 212, 210, 209, 208, 207, 205, 204, 203, 201, 200, 199, 198, 197, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 182, 181, 180, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 168, 167, 166, 165, 164, 163, 163, 162, 161, 160, 160, 159, 158, 157, 156, 156, 155, 154, 154, 153, 152, 151, 151, 150, 149, 149, 148, 147, 147, 146, 145, 145, 144, 144, 143, 142, 142, 141, 140, 140, 139, 139, 138, 138, 137, 136, 136, 135, 135, 134, 134, 133, 133, 132, 132, 131, 130, 130, 129, 129, 128, 127, 126, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 113, 112, 111, 110, 109, 108, 107, 107, 106, 105, 104, 103, 102, 102, 101, 100, 
99, 98, 98, 97, 96, 95, 95, 94, 93, 92, 92, 91, 90, 89, 89, 88, 87, 87, 86, 85, 85, 84, 83, 83, 82, 81, 81, 80, 79, 79, 78, 77, 77, 76, 75, 75, 74, 74, 73, 72, 72, 71, 70, 70, 69, 69, 68, 68, 67, 66, 66, 65, 65, 64, 64, 63, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48, 48, 47, 47, 46, 46, 45, 45, 44, 44, 44, 43, 43, 42, 42, 41, 41, 41, 40, 40, 39, 39, 38, 38, 38, 37, 37, 36, 36, 36, 35, 35, 34, 34, 34, 33, 33, 32, 32, 32, 31, 31, 30, 30, 30, 29, 29, 29, 28, 28, 27, 27, 27, 26, 26, 26, 25, 25, 25, 24, 24, 23, 23, 23, 22, 22, 22, 21, 21, 21, 20, 20, 20, 19, 19, 19, 18, 
18, 18, 17, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 
5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0];
//#[allow(clippy::excessive_precision)]
//const ENTROPY_LOOKUP: [f32; 512] = [13.0, 8.91253715874965, 7.955605880641543, 7.385290155884807, 6.977632186971531, 6.660149997115362, 6.400087157812884, 6.179821037584815, 5.988772744576749, 5.820090909985075, 5.669083121885373, 5.532394449916989, 5.407542962731906, 5.292640867919101, 5.186218808782963, 5.087110663770047, 4.994375450806124, 4.907242859080154, 4.825074317499332, 4.74733456754974, 4.673570512877699, 4.6033952188181475, 4.536475626728817, 4.472522993939605, 4.411285364417726, 4.352541573545071, 4.2960964265553265, 4.24177678527327, 4.1894283652588475, 4.138913094004592, 4.090106916229957, 4.042897958437719, 3.9971849843929554, 3.952876087885966, 3.909887580335704, 3.868143039391213, 3.82757249135453, 3.7881117054540026, 3.7497015820936745, 3.7122876204505397, 3.6758194533812656, 3.6402504396776743, 3.60553730538969, 3.5716398272957117, 3.53852055271385, 3.5061445507591844, 3.4744791909049413, 3.4434939453280835, 3.413160212038177, 3.383451156221002, 3.3543415675912907, 3.325807731854311, 3.2978273146344477, 3.2703792564469683, 3.2434436774759097, 3.217001791079593, 3.191035825080745, 3.165528950015772, 3.1404652136173468, 3.115829480891573, 3.0916073792262573, 3.0677852480316123, 3.0443500924716305, 3.021289540893637, 2.998591805607191, 2.976245646700589, 2.954240338617314, 2.932565639243478, 2.9112117612830883, 2.8901693457212136, 2.869429437194579, 2.848983461107759, 2.828823202348231, 2.8089407854683386, 2.7893286562143706, 2.76997956429437, 2.750886547286262, 2.732042915597162, 2.713442238392039, 2.695078330418328, 2.6769452396583593, 2.6590372357483094, 2.641348799107108, 2.6238746107238216, 2.6066095425560083, 2.5895486484960144, 2.5726871558650313, 2.5560204573987453, 2.539544103690367, 2.523253796060539, 2.507145379825256, 2.491214837935496, 2.4754582849641515, 2.4598719614178943, 2.444452228352931, 2.429195562275503, 2.4140985503092325, 2.399157885612699, 2.384370363031905, 2.369732874973236, 2.3552424074837353, 2.3408960365263396, 2.3266909244383083, 2.31262431656253, 2.2986935380413724, 2.284895990763833, 2.2712291504573394, 2.2576905639160083, 2.244277846357711, 2.230988678902981, 2.2178208061687887, 2.2047720339713233, 2.191840227131624, 2.179023307378764, 2.1663192513452647, 2.1537260886500977, 2.1412419000645837, 2.128864815756963, 2.116593013611634, 2.1044247176193127, 2.09235819633448, 2.0803917613967524, 2.0685237661132048, 2.0567526040984485, 2.0450767079696823, 2.0334945480942626, 2.022004631387035, 2.010605500155265, 1.9992957309887573, 1.9880739336931885, 1.976938750264658, 1.965888853903408, 1.9549229480650538, 1.9440397655477017, 1.9332380676130965, 1.9225166431404885, 1.911874307811788, 1.9013099033264136, 1.890822296644728, 1.8804103792588764, 1.8700730664896028, 1.8598092968081592, 1.8496180311821544, 1.8394982524443149, 1.8294489646833025, 1.8194691926555118, 1.80955798121718, 1.7997143947756984, 1.7899375167597722, 1.7802264491071205, 1.77058031176958, 1.760998242234421, 1.7514793950615735, 1.7420229414360255, 1.7326280687347264, 1.72329398010757, 1.7140198940717843, 1.7048050441193474, 1.6956486783367646, 1.68655005903694, 1.677508462402529, 1.6685231781403835, 1.6595935091467033, 1.6507187711825462, 1.6418982925591499, 1.6331314138329014, 1.6244174875094959, 1.6157558777569243, 1.6071459601271272, 1.5985871212858194, 1.5900787587503176, 1.5816202806350383, 1.5732111054044347, 1.5648506616330728, 1.5565383877726051, 1.5482737319254916, 1.5400561516250668, 1.5318851136219038, 1.523760093676146, 1.5156805763557293, 1.5076460548401347, 1.4996560307295965, 1.491710013859657, 1.4838075221206304, 1.475948081282181, 1.468131224822448, 1.4603564937619637, 1.4526234365019002, 1.4449316086666595, 1.4372805729506788, 1.4296698989691972, 1.4220991631130273, 1.4145679484070397, 1.407075844372364, 1.3996224468921532, 1.3922073580807846, 1.384830186156395, 1.3774905453166657, 1.3701880556177715, 1.3629223428563364, 1.3556930384544224, 1.3484997793472637, 1.3413422078739101, 1.3342199716705168, 1.3271327235662047, 1.3200801214815774, 1.3130618283296094, 1.3060775119189887, 1.2991268448597366, 1.2922095044711746, 1.285325172692001, 1.2784735359925237, 1.2716542852890285, 1.2648671158600555, 1.2581117272647477, 1.2513878232630193, 1.244695111737599, 1.2380333046179117, 1.2314021178056285, 1.2248012711020149, 1.218230488136868, 1.2116894962990818, 1.2051780266687784, 1.1986958139509745, 1.192242596410733, 1.185818115809723, 1.179422117344226, 1.1730543495845154, 1.166714564415515, 1.160402516978823, 1.1541179656159386, 1.1478606718127555, 1.1416304001452602, 1.1354269182263579, 1.1292499966538851, 1.123099408959692, 1.116974931559826, 1.1108763437057656, 1.104803427436662, 1.0987559675326237, 1.0927337514689304, 1.0867365693712387, 1.080764213971652, 1.0748164805657905, 1.0688931669706574, 1.0629940734834105, 1.0571190028409616, 1.051267760180391, 
//1.045440153000168, 1.0396359911221678, 1.0338550866543983, 1.028097253954531, 1.0223623095941257, 1.016650072323543, 1.0109603630376112, 1.0052930047419009, 0.9996478225197019, 0.994024643499615, 0.9884232968238249, 0.9828436136168898, 0.9772854269552378, 0.9717485718372094, 0.9662328851536125, 0.9607382056589425, 0.9552643739430938, 0.9498112324035627, 0.9443786252182917, 0.9389663983188861, 0.9335743993644453, 0.9282024777157896, 0.9228504844102652, 0.9175182721368991, 0.9122056952120979, 0.9069126095557789, 0.9016388726679029, 0.8963843436054577, 0.8911488829598437, 0.8859323528347391, 0.8807346168242134, 0.8755555399914139, 0.8703949888474941, 0.8652528313309873, 0.8601289367875143, 0.8550231759498914, 0.8499354209185281, 0.8448655451422422, 0.8398134233993357, 0.8347789317790557, 0.8297619476633531, 0.8247623497089623, 0.8197800178298097, 0.8148148331796687, 0.8098666781351982, 0.8049354362791803, 0.8000209923841014, 0.7951232323960081, 0.7902420434186055, 0.7853773136976653, 0.7805289326056439, 0.7756967906266112, 0.7708807793414066, 0.7660807914130218, 0.7612967205722754, 0.7565284616036693, 0.7517759103315269, 0.7470389636063149, 0.7423175192912339, 0.7376114762489818, 0.7329207343287717, 0.7282451943535242, 0.7235847581073216, 0.7189393283229795, 0.7143088086699076, 0.7096931037420976, 0.7050921190463459, 0.700505760990637, 0.6959339368727187, 0.6913765548688622, 0.6868335240228021, 0.6823047542348352, 0.6777901562511062, 0.673289641653042, 0.6688031228469682, 0.6643305130538973, 0.6598717262994139, 0.6554266774037969, 0.6509952819722571, 0.6465774563852861, 0.6421731177892255, 0.6377821840869212, 0.63340457392856, 0.6290402067026126, 0.6246890025269298, 0.6203508822399627, 0.6160257673921454, 0.6117135802373511, 0.6074142437245338, 0.6031276814894397, 0.598853817846503, 0.5945925777807959, 0.5903438869401474, 0.5861076716273586, 0.5818838587925101, 0.5776723760254412, 0.5734731515482703, 0.5692861142080625, 0.5651111934696047, 0.560948319408269, 0.5567974227029867, 0.552658434629322, 0.5485312870526404, 0.5444159124213871, 0.5403122437604533, 0.536220214664616, 0.5321397592921215, 0.5280708123583087, 0.5240133091293449, 0.5199671854160504, 0.5159323775678011, 0.5119088224665238, 0.5078964575207716, 0.5038952206598726, 0.49990505032819454, 0.49592588547942285, 0.49195766557099363, 0.48800033055854897, 0.4840538208904808, 0.4801180775025737, 0.47619304181267147, 0.4722786557154829, 0.46837486157737424, 0.4644816022313312, 0.46059882097188826, 0.4567264615501969, 0.45286446816913695, 0.4490127854784802, 0.4451713585701478, 
//0.44134013297349334, 0.43751905465068525, 0.43370806999212824, 0.429907125811944, 0.42611616934353036, 0.4223351482351555, 0.41856401054561587, 0.4148027047399765, 0.41105117968531824, 0.4073093846465886, 0.40357726928247273, 0.3998547836413314, 0.39614187815719504, 0.3924385036457926, 0.38874461130066373, 0.3850601526892638, 0.3813850797491961, 0.37771934478441693, 0.37406290046153856, 0.37041569980615147, 0.36677769619920997, 0.363148843373466, 0.3595290954099119, 0.3559184067343253, 0.352316732113799, 0.3487240266533601, 0.3451402457926065, 0.34156534530238436, 0.3379992812815189, 0.33444201015358677, 0.3308934886636983, 0.3273536738753726, 0.32382252316739646, 0.3202999942307514, 0.3167860450655724, 0.31328063397815314, 0.3097837195779618, 0.3062952607747149, 0.3028152167754847, 0.29934354708183264, 0.29588021148698196, 0.2924251700730278, 0.2889783832081636, 0.2855398115439729, 0.2821094160127192, 0.27868715782468845, 0.2752729984655539, 0.27186689969378364, 0.2684688235380577, 0.26507873229473944, 0.2616965885253615, 0.25832235505414136, 0.25495599496553556, 0.25159747160181567, 0.2482467485606641, 0.24490378969282034, 0.24156855909973274, 0.23824102113124468, 0.23492114038331474, 0.23160888169575244, 0.22830421014997782, 0.22500709106682493, 0.2217174900043507, 0.2184353727556722, 0.21516070534685028, 0.21189345403475457, 0.2086335853049956, 0.20538106586985805, 0.20213586266625216, 0.19889794285370854, 0.19566727381238397, 0.19244382314107966, 0.1892275586553005, 0.18601844838532244, 0.18281646057428702, 0.1796215636763172, 0.176433726354651, 0.17325291747979676, 0.17007910612771004, 0.16691226157799122, 0.16375235331209975, 0.1605993510115873, 0.15745322455635746, 0.15431394402293674, 0.1511814796827622, 0.14805580200050067, 0.1449368816323755, 0.1418246894245101, 0.13871919641129082, 0.13562037381376157, 0.13252819303800745, 0.12944262567358494, 0.12636364349194906, 0.12329121844490487, 0.12022532266307273, 0.11716592845437823, 0.11411300830254412, 0.11106653486560347, 0.1080264809744449, 0.10499281963133811, 0.10196552400851289, 0.0989445674467273, 0.09592992345386087, 0.09292156570351473, 0.0899194680336457, 0.08692360444518776, 0.08393394910070917, 0.08095047632306951, 0.07797316059410342, 0.07500197655330432, 0.07203689899653298, 0.06907790287473148, 0.0661249632926566, 0.06317805550762046, 0.060237154928246994, 0.05730223711324278, 0.054373277770174866, 0.0514502527542664, 0.04853313806720472, 0.045621909855952936, 0.04271654441158468, 0.039817018168125884, 0.036923307701405506, 0.034035389727916865, 0.031153241103705653, 0.02827683882324017, 0.02540616001832241, 0.022541181956995615, 0.019681882042455356, 0.016828237811992608, 0.013980226935924378, 0.011137827216550988, 0.008301016587120458, 0.005469773110789333, 0.002644074979619049];

#[derive(Debug, Clone)]
pub struct WriterCounter {
  /// Bytes that would be shifted out to date
  bits: usize,
}

#[derive(Debug, Clone)]
pub struct WriterRecorder {
  /// Storage for tokens
  storage: Vec<(u16, u16, u16)>,
  /// 128 * bits
  bytes: usize,
}

#[derive(Debug, Clone)]
pub struct WriterEncoder {
  /// A buffer for output bytes with their associated carry flags.
  precarry: Vec<u16>,
  /// The low end of the current range.
  low: ec_window,
}

#[derive(Clone)]
pub struct WriterCheckpoint {
  /// Byte length coded/recorded to date
  stream_bytes: usize,
  /// To be defined by backend
  backend_var: usize,
  /// Saved number of values in the current range.
  rng: u16,
  /// Saved number of bits of data in the current value.
  cnt: i16,
}

/// Constructor for a counting Writer
impl WriterCounter {
  #[inline]
  pub const fn new() -> WriterBase<WriterCounter> {
    WriterBase::new(WriterCounter { bits: 0 })
  }
}

/// Constructor for a recording Writer
impl WriterRecorder {
  #[inline]
  pub const fn new() -> WriterBase<WriterRecorder> {
    WriterBase::new(WriterRecorder { storage: Vec::new(), bytes: 0 })
  }
}

/// Constructor for a encoding Writer
impl WriterEncoder {
  #[inline]
  pub const fn new() -> WriterBase<WriterEncoder> {
    WriterBase::new(WriterEncoder { precarry: Vec::new(), low: 0 })
  }
}

/// The Counter stores nothing we write to it, it merely counts the
/// bit usage like in an Encoder for cost analysis.
impl StorageBackend for WriterBase<WriterCounter> {
  #[inline]
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    self.s.bits += unsafe { *ENTROPY_LOOKUP.get_unchecked(((fl >> EC_PROB_SHIFT) - (fh >> EC_PROB_SHIFT)) as usize) as usize };
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.bits / 128
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_bytes: self.s.bits / 128,
      backend_var: self.s.bits,
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  #[inline]
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.bits = checkpoint.backend_var;
  }
}

/// The Recorder does not produce a range-coded bitstream, but it
/// still tracks the range coding progress like in an Encoder, as it
/// neds to be able to report bit costs for RDO decisions.  It stores a
/// pair of mostly-computed range coding values per token recorded.
impl StorageBackend for WriterBase<WriterRecorder> {
  #[inline]
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (_l, r) = self.lr_compute(fl, fh, nms);
    let d = 16 - ILog::ilog(r);
    let mut s = self.cnt + (d as i16);

    self.s.bytes += (s >= 0) as usize + (s >= 8) as usize;
    s -= 8 * ((s >= 0) as i16 + (s >= 8) as i16);

    self.rng = r << d;
    self.cnt = s;
    self.s.storage.push((fl, fh, nms));
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.bytes * 8
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_bytes: self.s.bytes,
      backend_var: self.s.storage.len(),
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  #[inline]
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.bytes = checkpoint.stream_bytes;
    self.s.storage.truncate(checkpoint.backend_var);
  }
}

/// An Encoder produces an actual range-coded bitstream from passed in
/// tokens.  It does not retain any information about the coded
/// tokens, only the resulting bitstream, and so it cannot be replayed
/// (only checkpointed and rolled back).
impl StorageBackend for WriterBase<WriterEncoder> {
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (l, r) = self.lr_compute(fl, fh, nms);
    let mut low = l + self.s.low;
    let mut c = self.cnt;
    let d = 16 - ILog::ilog(r);
    let mut s = c + (d as i16);

    if s >= 0 {
      c += 16;
      let mut m = (1 << c) - 1;
      if s >= 8 {
        self.s.precarry.push((low >> c) as u16);
        low &= m;
        c -= 8;
        m >>= 8;
      }
      self.s.precarry.push((low >> c) as u16);
      s = c + (d as i16) - 24;
      low &= m;
    }
    self.s.low = low << d;
    self.rng = r << d;
    self.cnt = s;
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.precarry.len() * 8
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_bytes: self.s.precarry.len(),
      backend_var: self.s.low as usize,
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.low = checkpoint.backend_var as ec_window;
    self.s.precarry.truncate(checkpoint.stream_bytes);
  }
}

/// A few local helper functions needed by the Writer that are not
/// part of the public interface.
impl<S> WriterBase<S> {
  /// Internal constructor called by the subtypes that implement the
  /// actual encoder and Recorder.
  #[inline]
  #[cfg(not(feature = "desync_finder"))]
  const fn new(storage: S) -> Self {
    WriterBase { rng: 0x8000, cnt: -9, fake_bits_frac: 0, s: storage }
  }

  #[inline]
  #[cfg(feature = "desync_finder")]
  fn new(storage: S) -> Self {
    WriterBase {
      rng: 0x8000,
      cnt: -9,
      debug: std::env::var_os("RAV1E_DEBUG").is_some(),
      fake_bits_frac: 0,
      s: storage,
    }
  }

  /// Compute low and range values from token cdf values and local state
  fn lr_compute(&mut self, fl: u16, fh: u16, nms: u16) -> (ec_window, u16) {
    let u: u32;
    let v: u32;
    let mut r = self.rng as u32;
    debug_assert!(32768 <= r);
    if fl < 32768 {
      u = (((r >> 8) * (fl as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB * nms as u32;
      v = (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB * (nms - 1) as u32;
      (r - u, (u - v) as u16)
    } else {
      r -= (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB * (nms - 1) as u32;
      (0, r as u16)
    }
  }

  /// Given the current total integer number of bits used and the current value of
  /// rng, computes the fraction number of bits used to `OD_BITRES` precision.
  /// This is used by `od_ec_enc_tell_frac()` and `od_ec_dec_tell_frac()`.
  /// `nbits_total`: The number of whole bits currently used, i.e., the value
  ///                returned by `od_ec_enc_tell()` or `od_ec_dec_tell()`.
  /// `rng`: The current value of rng from either the encoder or decoder state.
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///         rounding error is in the positive direction).
  fn frac_compute(nbits_total: u32, mut rng: u32) -> u32 {
    // To handle the non-integral number of bits still left in the encoder/decoder
    //  state, we compute the worst-case number of bits of val that must be
    //  encoded to ensure that the value is inside the range for any possible
    //  subsequent bits.
    // The computation here is independent of val itself (the decoder does not
    //  even track that value), even though the real number of bits used after
    //  od_ec_enc_done() may be 1 smaller if rng is a power of two and the
    //  corresponding trailing bits of val are all zeros.
    // If we did try to track that special case, then coding a value with a
    //  probability of 1/(1 << n) might sometimes appear to use more than n bits.
    // This may help explain the surprising result that a newly initialized
    //  encoder or decoder claims to have used 1 bit.
    let nbits = nbits_total << OD_BITRES;
    let mut l = 0;
    for _ in 0..OD_BITRES {
      rng = (rng * rng) >> 15;
      let b = rng >> 16;
      l = (l << 1) | b;
      rng >>= b;
    }
    nbits - l
  }

  const fn recenter(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
      v
    } else if v >= r {
      (v - r) << 1
    } else {
      ((r - v) << 1) - 1
    }
  }

  #[cfg(feature = "desync_finder")]
  fn print_backtrace(&self, s: u32) {
    let mut depth = 3;
    backtrace::trace(|frame| {
      let ip = frame.ip();

      depth -= 1;

      if depth == 0 {
        backtrace::resolve(ip, |symbol| {
          if let Some(name) = symbol.name() {
            println!("Writing symbol {} from {}", s, name);
          }
        });
        false
      } else {
        true
      }
    });
  }
}

/// Replay implementation specific to the Recorder
impl WriterBase<WriterRecorder> {
  /// Replays the partially-computed range tokens out of the Recorder's
  /// storage and into the passed in Writer, which may be an Encoder
  /// or another Recorder.  Clears the Recorder after replay.
  pub fn replay(&mut self, dest: &mut dyn StorageBackend) {
    for &(fl, fh, nms) in &self.s.storage {
      dest.store(fl, fh, nms);
    }
    self.rng = 0x8000;
    self.cnt = -9;
    self.s.storage.truncate(0);
    self.s.bytes = 0;
  }
}

/// Done implementation specific to the Encoder
impl WriterBase<WriterEncoder> {
  /// Indicates that there are no more symbols to encode.  Flushes
  /// remaining state into coding and returns a vector containing the
  /// final bitstream.
  pub fn done(&mut self) -> Vec<u8> {
    // We output the minimum number of bits that ensures that the symbols encoded
    // thus far will be decoded correctly regardless of the bits that follow.
    let l = self.s.low;
    let mut c = self.cnt;
    let mut s = 10;
    let m = 0x3FFF;
    let mut e = ((l + m) & !m) | (m + 1);

    s += c;

    if s > 0 {
      let mut n = (1 << (c + 16)) - 1;

      loop {
        self.s.precarry.push((e >> (c + 16)) as u16);
        e &= n;
        s -= 8;
        c -= 8;
        n >>= 8;

        if s <= 0 {
          break;
        }
      }
    }

    let mut c = 0;
    let mut offs = self.s.precarry.len();
    // dynamic allocation: grows during encode
    let mut out = vec![0_u8; offs];
    while offs > 0 {
      offs -= 1;
      c += self.s.precarry[offs];
      out[offs] = c as u8;
      c >>= 8;
    }

    out
  }
}

/// Generic/shared implementation for `Writer`s with `StorageBackend`s
/// (ie, `Encoder`s and `Recorder`s)
impl<S> Writer for WriterBase<S>
where
  WriterBase<S>: StorageBackend,
{
  /// Encode a single binary value.
  /// `val`: The value to encode (0 or 1).
  /// `f`: The probability that the val is one, scaled by 32768.
  fn bool(&mut self, val: bool, f: u16) {
    debug_assert!(0 < f);
    debug_assert!(f < 32768);
    self.symbol(u32::from(val), &[f, 0]);
  }
  /// Encode a single boolean value.
  ///
  /// - `val`: The value to encode (`false` or `true`).
  /// - `f`: The probability that the `val` is `true`, scaled by `32768`.
  fn bit(&mut self, bit: u16) {
    self.bool(bit == 1, 16384);
  }
  // fake add bits
  fn add_bits_frac(&mut self, bits_frac: u32) {
    self.fake_bits_frac += bits_frac
  }
  /// Encode a literal bitstring, bit by bit in MSB order, with flat
  /// probability.
  ///
  /// - 'bits': Length of bitstring
  /// - 's': Bit string to encode
  fn literal(&mut self, bits: u8, s: u32) {
    for bit in (0..bits).rev() {
      self.bit((1 & (s >> bit)) as u16);
    }
  }
  /// Encodes a symbol given a cumulative distribution function (CDF) table in Q15.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///       The values must be monotonically non-decreasing, and the last value
  ///       must be greater than 32704. There should be at most 16 values.
  ///       The lower 6 bits of the last value hold the count.
  #[inline(always)]
  fn symbol<const CDF_LEN: usize>(&mut self, s: u32, cdf: &[u16; CDF_LEN]) {
    debug_assert!(cdf[cdf.len() - 1] < (1 << EC_PROB_SHIFT));
    let s = s as usize;
    debug_assert!(s < cdf.len());
    // The above is stricter than the following overflow check: s <= cdf.len()
    let nms = cdf.len() - s;
    let fl = if s > 0 {
      // SAFETY: We asserted that s is less than the length of the cdf
      unsafe { *cdf.get_unchecked(s - 1) }
    } else {
      32768
    };
    // SAFETY: We asserted that s is less than the length of the cdf
    let fh = unsafe { *cdf.get_unchecked(s) };
    debug_assert!((fh >> EC_PROB_SHIFT) <= (fl >> EC_PROB_SHIFT));
    debug_assert!(fl <= 32768);
    self.store(fl, fh, nms as u16);
  }
  /// Encodes a symbol given a cumulative distribution function (CDF)
  /// table in Q15, then updates the CDF probabilities to reflect we've
  /// written one more symbol 's'.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///       The values must be monotonically non-decreasing, and the last value
  ///       must be greater 32704. There should be at most 16 values.
  ///       The lower 6 bits of the last value hold the count.
  fn symbol_with_update<const CDF_LEN: usize>(
    &mut self, s: u32, cdf: &mut [u16; CDF_LEN], log: &mut CDFContextLog,
  ) {
    #[cfg(feature = "desync_finder")]
    {
      if self.debug {
        self.print_backtrace(s);
      }
    }
    log.push(cdf);
    self.symbol(s, cdf);

    update_cdf(cdf, s);
  }
  /// Returns approximate cost for a symbol given a cumulative
  /// distribution function (CDF) table and current write state.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///       The values must be monotonically non-decreasing, and the last value
  ///       must be greater than 32704. There should be at most 16 values.
  ///       The lower 6 bits of the last value hold the count.
  fn symbol_bits(&self, s: u32, cdf: &[u16]) -> u32 {
    let mut bits = 0;
    debug_assert!(cdf[cdf.len() - 1] < (1 << EC_PROB_SHIFT));
    debug_assert!(32768 <= self.rng);
    let rng = (self.rng >> 8) as u32;
    let fh = cdf[s as usize] as u32 >> EC_PROB_SHIFT;
    let r = if s > 0 {
      let fl = cdf[s as usize - 1] as u32 >> EC_PROB_SHIFT;
      ((rng * fl) >> (7 - EC_PROB_SHIFT)) - ((rng * fh) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB
    } else {
      let nms1 = cdf.len() as u32 - s - 1;
      self.rng as u32
        - ((rng * fh) >> (7 - EC_PROB_SHIFT))
        - nms1 * EC_MIN_PROB
    };

    // The 9 here counteracts the offset of -9 baked into cnt.  Don't include a termination bit.
    let pre = Self::frac_compute((self.cnt + 9) as u32, self.rng as u32);
    let d = 16 - ILog::ilog(r);
    let mut c = self.cnt;
    let mut sh = c + (d as i16);
    if sh >= 0 {
      c += 16;
      if sh >= 8 {
        bits += 8;
        c -= 8;
      }
      bits += 8;
      sh = c + (d as i16) - 24;
    }
    // The 9 here counteracts the offset of -9 baked into cnt.  Don't include a termination bit.
    Self::frac_compute((bits + sh + 9) as u32, r << d) - pre
  }
  /// Encode a golomb to the bitstream.
  ///
  /// - 'level': passed in value to encode
  fn write_golomb(&mut self, level: u32) {
    let x = level + 1;
    let length = 32 - x.leading_zeros();

    for _ in 0..length - 1 {
      self.bit(0);
    }

    for i in (0..length).rev() {
      self.bit(((x >> i) & 0x01) as u16);
    }
  }
  /// Write a value `v` in `[0, n-1]` quasi-uniformly
  /// - `n`: size of interval
  /// - `v`: value to encode
  fn write_quniform(&mut self, n: u32, v: u32) {
    if n > 1 {
      let l = msb(n as i32) as u8 + 1;
      let m = (1 << l) - n;
      if v < m {
        self.literal(l - 1, v);
      } else {
        self.literal(l - 1, m + ((v - m) >> 1));
        self.literal(1, (v - m) & 1);
      }
    }
  }
  /// Returns `QOD_BITRES` bits for a value `v` in `[0, n-1]` quasi-uniformly
  /// - `n`: size of interval
  /// - `v`: value to encode
  fn count_quniform(&self, n: u32, v: u32) -> u32 {
    let mut bits = 0;
    if n > 1 {
      let l = (msb(n as i32) + 1) as u32;
      let m = (1 << l) - n;
      bits += (l - 1) << OD_BITRES;
      if v >= m {
        bits += 1 << OD_BITRES;
      }
    }
    bits
  }
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite subexponential
  ///
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `v`: value to encode
  fn write_subexp(&mut self, n: u32, k: u8, v: u32) {
    let mut i = 0;
    let mut mk = 0;
    loop {
      let b = if i != 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        self.write_quniform(n - mk, v - mk);
        break;
      } else {
        let t = v >= mk + a;
        self.bool(t, 16384);
        if t {
          i += 1;
          mk += a;
        } else {
          self.literal(b, v - mk);
          break;
        }
      }
    }
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[0, n-1]` with parameter `k`
  /// as finite subexponential
  ///
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `v`: value to encode
  fn count_subexp(&self, n: u32, k: u8, v: u32) -> u32 {
    let mut i = 0;
    let mut mk = 0;
    let mut bits = 0;
    loop {
      let b = if i != 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        bits += self.count_quniform(n - mk, v - mk);
        break;
      } else {
        let t = v >= mk + a;
        bits += 1 << OD_BITRES;
        if t {
          i += 1;
          mk += a;
        } else {
          bits += (b as u32) << OD_BITRES;
          break;
        }
      }
    }
    bits
  }
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[0, n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn write_unsigned_subexp_with_ref(&mut self, v: u32, n: u32, k: u8, r: u32) {
    if (r << 1) <= n {
      self.write_subexp(n, k, Self::recenter(r, v));
    } else {
      self.write_subexp(n, k, Self::recenter(n - 1 - r, n - 1 - v));
    }
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[0, n-1]`
  /// with parameter `k` as finite subexponential based on a
  /// reference `r` also in `[0, n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn count_unsigned_subexp_with_ref(
    &self, v: u32, n: u32, k: u8, r: u32,
  ) -> u32 {
    if (r << 1) <= n {
      self.count_subexp(n, k, Self::recenter(r, v))
    } else {
      self.count_subexp(n, k, Self::recenter(n - 1 - r, n - 1 - v))
    }
  }
  /// Write symbol `v` in `[-(n-1), n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[-(n-1), n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn write_signed_subexp_with_ref(
    &mut self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) {
    self.write_unsigned_subexp_with_ref(
      (v - low) as u32,
      (high - low) as u32,
      k,
      (r - low) as u32,
    );
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[-(n-1), n-1]`
  /// with parameter `k` as finite subexponential based on a
  /// reference `r` also in `[-(n-1), n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference

  fn count_signed_subexp_with_ref(
    &self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) -> u32 {
    self.count_unsigned_subexp_with_ref(
      (v - low) as u32,
      (high - low) as u32,
      k,
      (r - low) as u32,
    )
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions.  The value
  /// will be the same whether using an `Encoder` or `Recorder`.
  ///
  /// Return: The integer number of bits.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell(&mut self) -> u32 {
    // The 10 here counteracts the offset of -9 baked into cnt, and adds 1 extra
    // bit, which we reserve for terminating the stream.
    (((self.stream_bits()) as i32) + (self.cnt as i32) + 10) as u32
      + (self.fake_bits_frac >> 8)
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions. The value
  /// will be the same whether using an `Encoder` or `Recorder`.
  ///
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell_frac(&mut self) -> u32 {
    Self::frac_compute(self.tell(), self.rng as u32) + self.fake_bits_frac
  }
  /// Save current point in coding/recording to a checkpoint that can
  /// be restored later.  A `WriterCheckpoint` can be generated for an
  /// `Encoder` or `Recorder`, but can only be used to rollback the `Writer`
  /// instance from which it was generated.
  fn checkpoint(&mut self) -> WriterCheckpoint {
    StorageBackend::checkpoint(self)
  }
  /// Roll back a given `Writer` to the state saved in the `WriterCheckpoint`
  ///
  /// - 'wc': Saved `Writer` state/posiiton to restore
  fn rollback(&mut self, wc: &WriterCheckpoint) {
    StorageBackend::rollback(self, wc)
  }
}

pub trait BCodeWriter {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16;
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16,
  ) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error>;
}

impl<W: io::Write> BCodeWriter for BitWriter<W, BigEndian> {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v around a reference r */
    if v > (r << 1) {
      v
    } else if v >= r {
      (v - r) << 1
    } else {
      ((r - v) << 1) - 1
    }
  }
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v in [0, n-1] around a
    reference r also in [0, n-1] */
    if (r << 1) <= n {
      self.recenter_nonneg(r, v)
    } else {
      self.recenter_nonneg(n - 1 - r, n - 1 - v)
    }
  }
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error> {
    if n > 1 {
      let l = msb(n as i32) as u8 + 1;
      let m = (1 << l) - n;
      if v < m {
        self.write(l as u32 - 1, v)
      } else {
        self.write(l as u32 - 1, m + ((v - m) >> 1))?;
        self.write(1, (v - m) & 1)
      }
    } else {
      Ok(())
    }
  }
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16,
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with parameter k */
    let mut i = 0;
    let mut mk = 0;
    loop {
      let b = if i > 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        return self.write_quniform(n - mk, v - mk);
      } else {
        let t = v >= mk + a;
        self.write_bit(t)?;
        if t {
          i += 1;
          mk += a;
        } else {
          return self.write(b as u32, v - mk);
        }
      }
    }
  }
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with
    parameter k based on a reference ref also in [0, n-1].
    Recenters symbol around r first and then uses a finite subexponential code. */
    let recentered_v = self.recenter_finite_nonneg(n, r as u16, v as u16);
    self.write_subexpfin(n, k, recentered_v)
  }
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error> {
    /* Signed version of the above function */
    self.write_refsubexpfin(
      (n << 1) - 1,
      k,
      r + (n - 1) as i16,
      v + (n - 1) as i16,
    )
  }
}

pub(crate) fn cdf_to_pdf<const CDF_LEN: usize>(
  cdf: &[u16; CDF_LEN],
) -> [u16; CDF_LEN] {
  let mut pdf = [0; CDF_LEN];
  let mut z = 32768u16 >> EC_PROB_SHIFT;
  for (d, &a) in pdf.iter_mut().zip(cdf.iter()) {
    *d = z - (a >> EC_PROB_SHIFT);
    z = a >> EC_PROB_SHIFT;
  }
  pdf
}

pub(crate) mod rust {
  // Function to update the CDF for Writer calls that do so.
  #[inline]
  pub fn update_cdf<const N: usize>(cdf: &mut [u16; N], val: u32) {
    use crate::context::CDF_LEN_MAX;
    let nsymbs = cdf.len();
    let mut rate = 3 + (nsymbs >> 1).min(2);
    if let Some(count) = cdf.last_mut() {
      rate += (*count >> 4) as usize;
      *count += 1 - (*count >> 5);
    } else {
      return;
    }
    // Single loop (faster)
    for (i, v) in
      cdf[..nsymbs - 1].iter_mut().enumerate().take(CDF_LEN_MAX - 1)
    {
      if i as u32 >= val {
        *v -= *v >> rate;
      } else {
        *v += (32768 - *v) >> rate;
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  const WINDOW_SIZE: i16 = 32;
  const LOTS_OF_BITS: i16 = 0x4000;

  #[derive(Debug)]
  struct Reader<'a> {
    buf: &'a [u8],
    bptr: usize,
    dif: ec_window,
    rng: u16,
    cnt: i16,
  }

  impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
      let mut r = Reader {
        buf,
        bptr: 0,
        dif: (1 << (WINDOW_SIZE - 1)) - 1,
        rng: 0x8000,
        cnt: -15,
      };
      r.refill();
      r
    }

    fn refill(&mut self) {
      let mut s = WINDOW_SIZE - 9 - (self.cnt + 15);
      while s >= 0 && self.bptr < self.buf.len() {
        assert!(s <= WINDOW_SIZE - 8);
        self.dif ^= (self.buf[self.bptr] as ec_window) << s;
        self.cnt += 8;
        s -= 8;
        self.bptr += 1;
      }
      if self.bptr >= self.buf.len() {
        self.cnt = LOTS_OF_BITS;
      }
    }

    fn normalize(&mut self, dif: ec_window, rng: u32) {
      assert!(rng <= 65536);
      let d = rng.leading_zeros() - 16;
      //let d = 16 - (32-rng.leading_zeros());
      //msb(rng) = 31-rng.leading_zeros();
      self.cnt -= d as i16;
      /*This is equivalent to shifting in 1's instead of 0's.*/
      self.dif = ((dif + 1) << d) - 1;
      self.rng = (rng << d) as u16;
      if self.cnt < 0 {
        self.refill()
      }
    }

    fn bool(&mut self, f: u32) -> bool {
      assert!(f < 32768);
      let r = self.rng as u32;
      assert!(self.dif >> (WINDOW_SIZE - 16) < r);
      assert!(32768 <= r);
      let v = (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB;
      let vw = v << (WINDOW_SIZE - 16);
      let (dif, rng, ret) = if self.dif >= vw {
        (self.dif - vw, r - v, false)
      } else {
        (self.dif, v, true)
      };
      self.normalize(dif, rng);
      ret
    }

    fn symbol(&mut self, icdf: &[u16]) -> i32 {
      let r = self.rng as u32;
      assert!(self.dif >> (WINDOW_SIZE - 16) < r);
      assert!(32768 <= r);
      let n = icdf.len() as u32 - 1;
      let c = self.dif >> (WINDOW_SIZE - 16);
      let mut v = self.rng as u32;
      let mut ret = 0i32;
      let mut u = v;
      v = ((r >> 8) * (icdf[ret as usize] as u32 >> EC_PROB_SHIFT))
        >> (7 - EC_PROB_SHIFT);
      v += EC_MIN_PROB * (n - ret as u32);
      while c < v {
        u = v;
        ret += 1;
        v = ((r >> 8) * (icdf[ret as usize] as u32 >> EC_PROB_SHIFT))
          >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB * (n - ret as u32);
      }
      assert!(v < u);
      assert!(u <= r);
      let new_dif = self.dif - (v << (WINDOW_SIZE - 16));
      self.normalize(new_dif, u - v);
      ret
    }
  }

  #[test]
  fn booleans() {
    let mut w = WriterEncoder::new();

    w.bool(false, 1);
    w.bool(true, 2);
    w.bool(false, 3);
    w.bool(true, 1);
    w.bool(true, 2);
    w.bool(false, 3);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert!(!r.bool(1));
    assert!(r.bool(2));
    assert!(!r.bool(3));
    assert!(r.bool(1));
    assert!(r.bool(2));
    assert!(!r.bool(3));
  }

  #[test]
  fn cdf() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }

  #[test]
  fn mixed() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(1, &cdf);
    w.bool(true, 1);
    w.symbol(1, &cdf);
    w.bool(false, 2);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 1);
    assert!(r.bool(1));
    assert_eq!(r.symbol(&cdf), 1);
    assert!(!r.bool(2));
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }
}
