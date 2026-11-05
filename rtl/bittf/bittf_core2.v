// BitTF jadro v2 (vykonove): stejna ISA v1 a chovani jako bittf_core (v1), ale
//  - pamet pres NP = 5 radkovych portu (LB bajtu): 0 = instrukce, skalary a zapis vysledku,
//    1/2/3 = operandy A/B/C, 4 = indexy gatheru; na FPGA kazdy port = kanal HBM,
//  - EW a RED ve VL drahach (souvisle nebo broadcast operandy), jinak po 1 prvku,
//  - MM na poli TM x TN (bittf_mm).
// Port: pozadavek (valid/ready, zapis s bajtovou maskou), odpovedi na cteni v poradi, wack = potvrzeni zapisu.
module bittf_core2 #(
    parameter VL = 8,
    parameter LB = 64,
    parameter TM = 32,
    parameter TN = 32,
    parameter KC = 64,
    parameter LVL = 32
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire [31:0]       prog_base,
    output wire              busy,
    output reg               done,
    output reg  [7:0]        err,
    input  wire              csr_we,
    input  wire [1:0]        csr_addr,
    input  wire [31:0]       csr_wdata,
    output reg  [31:0]       csr_rdata,
    // radkove porty (vektory po portech)
    output wire [4:0]        p_rq_valid,
    input  wire [4:0]        p_rq_ready,
    output wire [4:0]        p_rq_we,
    output wire [5*26-1:0]   p_rq_line,
    output wire [5*LB*8-1:0] p_rq_wdata,
    output wire [5*LB-1:0]   p_rq_wstrb,
    input  wire [4:0]        p_rs_valid,
    input  wire [5*LB*8-1:0] p_rs_data,
    input  wire [4:0]        p_wack,
    // ladeni
    output reg               dbg_instr_done,
    output reg  [31:0]       dbg_pc,
    input  wire [5:0]        dbg_ridx,
    output wire [63:0]       dbg_rdata
);
`include "bittf_isa.vh"
`include "bittf_func.vh"

    localparam ERR_FIT = 8'd1, ERR_OP = 8'd3;
    localparam LBB = $clog2(LB);

    localparam S_IDLE = 4'd0, S_FETCH = 4'd1, S_DEC = 4'd2, S_SCAL = 4'd3, S_SLD = 4'd4, S_SST = 4'd5,
               S_EW = 4'd6, S_RED = 4'd7, S_MM = 4'd8, S_DRAIN = 4'd9, S_END = 4'd10, S_HALT = 4'd11,
               S_ERR = 4'd12, S_ALUW = 4'd13, S_SETUP = 4'd14;
    reg [3:0] st;
    assign busy = st != S_IDLE;

    reg [31:0] ir [0:47];
    reg [31:0] pc, base;
    reg [63:0] R [0:63];
    reg [31:0] csr_seed, csr_ctr, csr_cyc;
    assign dbg_rdata = R[dbg_ridx];
    integer k, l;

    always @(*) begin
        case (csr_addr)
            2'd0: csr_rdata = csr_seed;
            2'd1: csr_rdata = csr_ctr;
            2'd2: csr_rdata = {24'd0, err};
            default: csr_rdata = csr_cyc;
        endcase
    end

    // ------------------------------------------------------------ dekodovani
    wire [7:0]  op = ir[0][7:0];
    wire [7:0]  fn = ir[0][15:8];
    wire [15:0] fl = ir[0][31:16];
    wire [5:0]  rs1 = ir[36][5:0], rs2 = ir[36][13:8], rd = ir[36][21:16], rs3 = ir[36][29:24];
    wire [31:0] imm = ir[37], imm3 = ir[39];
    wire signed [63:0] simm = {{32{imm[31]}}, imm};
    wire signed [63:0] simm2 = {{32{ir[38][31]}}, ir[38]};
    wire signed [63:0] vR1 = R[rs1], vR2 = R[rs2], vR3 = R[rs3];
    function [31:0] obase(input integer x); obase = ir[5 + 6 * x]; endfunction
    function [1:0]  olw(input integer x);   olw = ir[6 + 6 * x][1:0]; endfunction
    function        ouns(input integer x);  ouns = ir[6 + 6 * x][2]; endfunction
    function        ogath(input integer x); ogath = ir[6 + 6 * x][3]; endfunction
    function [2:0]  omode(input integer x); omode = ir[6 + 6 * x][6:4]; endfunction
    function signed [63:0] ostr(input integer x, input integer kk);
        ostr = {{32{ir[7 + 6 * x + kk][31]}}, ir[7 + 6 * x + kk]};
    endfunction
    function [255:0] ostrv(input integer x);
        ostrv = {ostr(x, 3), ostr(x, 2), ostr(x, 1), ostr(x, 0)};
    endfunction
    wire [127:0] cntv = {ir[4], ir[3], ir[2], ir[1]};
    wire [255:0] gstrv = {{{32{ir[34][31]}}, ir[34]}, {{32{ir[33][31]}}, ir[33]}, {{32{ir[32][31]}}, ir[32]},
                          {{32{ir[31][31]}}, ir[31]}};
    wire signed [63:0] gscale = {{32{ir[35][31]}}, ir[35]};

    wire is_ew = op == OP_EW, is_red = op == OP_RED, is_mm = op == OP_MM;
    wire [2:0] tc = fl[2:0];
    wire [2:0] rpre = fl[10:8];
    wire need_a = is_ew ? EFN_NEED_A[fn[4:0]] : rpre != RP_SH;
    wire need_b = is_ew ? EFN_NEED_B[fn[4:0]] : (rpre == RP_AB || rpre == RP_ABC);
    wire need_c = is_ew ? (EFN_NEED_C[fn[4:0]] || tc != 0) : (rpre == RP_ABC || (rpre == RP_SH && tc != 0));
    wire uses_rng = is_ew && EFN_RNGMASK[fn[4:0]];
    wire is_div = is_ew && (fn == E_RCP || fn == E_MEANDIV || fn == E_MODU);
    wire any_gath = ogath(0) | ogath(1) | ogath(2) | ogath(3);
    function s01(input integer x); s01 = ostr(x, 0) == 0 || ostr(x, 0) == 1; endfunction
    wire vec_ok = !any_gath && !is_div && (!need_a || s01(1)) && (!need_b || s01(2)) && (!need_c || s01(3)) &&
                  (is_red || ostr(0, 0) == 1);
    reg vec;

    // ------------------------------------------------------------ porty
    // port 0: rizeni (fetch, skalary) nebo zapisovac
    reg         c_rq_valid, c_rq_we;
    reg  [25:0] c_rq_line;
    reg  [LB*8-1:0] c_rq_wdata;
    reg  [LB-1:0]   c_rq_wstrb;
    wire        w_rq_valid;
    wire [25:0] w_rq_line;
    wire [LB*8-1:0] w_rq_wdata;
    wire [LB-1:0]   w_rq_wstrb;
    wire use_w = st == S_EW || st == S_RED || st == S_MM || st == S_DRAIN || st == S_ALUW;
    assign p_rq_valid[0] = use_w ? w_rq_valid : c_rq_valid;
    assign p_rq_we[0]    = use_w ? 1'b1 : c_rq_we;
    assign p_rq_line[25:0] = use_w ? w_rq_line : c_rq_line;
    assign p_rq_wdata[LB*8-1:0] = use_w ? w_rq_wdata : c_rq_wdata;
    assign p_rq_wstrb[LB-1:0] = use_w ? w_rq_wstrb : c_rq_wstrb;
    wire [LB*8-1:0] rs0 = p_rs_data[LB*8-1:0];

    // ------------------------------------------------------------ proudy EW/RED
    reg  s_start;
    wire [4:0] ov;              // A B C G (indexy 1..4)
    wire [VL*64-1:0] od1, od2, od3, od4;
    wire [6:0] on1, on2, on3, on4;
    wire [4:0] sbusy;
    wire [4:0] oready;
    wire g_ready1, g_ready2, g_ready3;
    wire [25:0] l1, l2, l3, l4;
    wire v1, v2, v3, v4;
    // ktery operand dostava gather
    wire gath_rdy = ogath(1) ? g_ready1 : ogath(2) ? g_ready2 : ogath(3) ? g_ready3 : 1'b0;
    wire signed [63:0] gval = od4[63:0];

    bittf_stream #(.VL(VL), .LB(LB)) sA (
        .clk(clk), .rst(rst), .start(s_start && need_a), .base(obase(1)), .lw(olw(1)), .uns(ouns(1)), .vec(vec),
        .cnt(cntv), .str(ostrv(1)), .gath(ogath(1)), .gscale(gscale), .gidx_valid(ov[4]), .gidx_ready(g_ready1),
        .gidx(gval), .rq_valid(v1), .rq_ready(p_rq_ready[1]), .rq_line(l1), .rs_valid(p_rs_valid[1] && !is_mm),
        .rs_data(p_rs_data[LB*8 +: LB*8]), .o_valid(ov[1]), .o_ready(oready[1]), .o_data(od1), .o_n(on1), .busy(sbusy[1]));
    bittf_stream #(.VL(VL), .LB(LB)) sB (
        .clk(clk), .rst(rst), .start(s_start && need_b), .base(obase(2)), .lw(olw(2)), .uns(ouns(2)), .vec(vec),
        .cnt(cntv), .str(ostrv(2)), .gath(ogath(2)), .gscale(gscale), .gidx_valid(ov[4]), .gidx_ready(g_ready2),
        .gidx(gval), .rq_valid(v2), .rq_ready(p_rq_ready[2]), .rq_line(l2), .rs_valid(p_rs_valid[2] && !is_mm),
        .rs_data(p_rs_data[2*LB*8 +: LB*8]), .o_valid(ov[2]), .o_ready(oready[2]), .o_data(od2), .o_n(on2), .busy(sbusy[2]));
    bittf_stream #(.VL(VL), .LB(LB)) sC (
        .clk(clk), .rst(rst), .start(s_start && need_c), .base(obase(3)), .lw(olw(3)), .uns(ouns(3)), .vec(vec),
        .cnt(cntv), .str(ostrv(3)), .gath(ogath(3)), .gscale(gscale), .gidx_valid(ov[4]), .gidx_ready(g_ready3),
        .gidx(gval), .rq_valid(v3), .rq_ready(p_rq_ready[3]), .rq_line(l3), .rs_valid(p_rs_valid[3]),
        .rs_data(p_rs_data[3*LB*8 +: LB*8]), .o_valid(ov[3]), .o_ready(oready[3]), .o_data(od3), .o_n(on3), .busy(sbusy[3]));
    bittf_stream #(.VL(VL), .LB(LB)) sG (
        .clk(clk), .rst(rst), .start(s_start && any_gath), .base(ir[29]), .lw(ir[30][1:0]), .uns(ir[30][2]), .vec(1'b0),
        .cnt(cntv), .str(gstrv), .gath(1'b0), .gscale(64'sd0), .gidx_valid(1'b0), .gidx_ready(),
        .gidx(64'sd0), .rq_valid(v4), .rq_ready(p_rq_ready[4]), .rq_line(l4), .rs_valid(p_rs_valid[4]),
        .rs_data(p_rs_data[4*LB*8 +: LB*8]), .o_valid(ov[4]), .o_ready(gath_rdy), .o_data(od4), .o_n(on4), .busy(sbusy[4]));
    assign ov[0] = 1'b0;
    assign sbusy[0] = 1'b0;
    assign oready[0] = 1'b0;

    // MM
    reg  mm_start;
    wire mm_ra_v, mm_rb_v, mm_done, mm_err;
    wire [25:0] mm_ra_l, mm_rb_l;
    wire mm_w_start, mm_w_vec, mm_w_valid;
    wire [31:0] mm_w_base;
    wire [127:0] mm_w_cnt;
    wire [255:0] mm_w_str;
    wire [VL*64-1:0] mm_w_data;
    wire w_ready;
    bittf_mm #(.TM(TM), .TN(TN), .KC(KC), .LVL(LVL), .VL(VL), .LB(LB)) mm (
        .clk(clk), .rst(rst), .start(mm_start), .n0(ir[1]), .n1(ir[2]), .n2(ir[3]), .nk(ir[4]),
        .a_base(obase(1)), .b_base(obase(2)), .d_base(obase(0)), .a_lw(olw(1)), .b_lw(olw(2)), .d_lw(olw(0)),
        .a_uns(ouns(1)), .b_uns(ouns(2)), .d_uns(ouns(0)), .b_sign(omode(2) == M_SIGN),
        .as1(ostr(1, 1)), .as2(ostr(1, 2)), .as3(ostr(1, 3)), .bs0(ostr(2, 0)), .bs2(ostr(2, 2)), .bs3(ostr(2, 3)),
        .ds0(ostr(0, 0)), .ds1(ostr(0, 1)), .ds2(ostr(0, 2)),
        .ra_valid(mm_ra_v), .rb_valid(mm_rb_v), .ra_ready(p_rq_ready[1]), .rb_ready(p_rq_ready[2]),
        .ra_line(mm_ra_l), .rb_line(mm_rb_l), .sa_valid(p_rs_valid[1] && is_mm), .sb_valid(p_rs_valid[2] && is_mm),
        .sa_data(p_rs_data[LB*8 +: LB*8]), .sb_data(p_rs_data[2*LB*8 +: LB*8]),
        .w_start(mm_w_start), .w_base(mm_w_base), .w_cnt(mm_w_cnt), .w_str(mm_w_str), .w_vec(mm_w_vec),
        .w_valid(mm_w_valid), .w_ready(w_ready), .w_data(mm_w_data), .done(mm_done), .err_fit(mm_err));

    assign p_rq_valid[1] = is_mm ? mm_ra_v : v1;
    assign p_rq_line[26 +: 26] = is_mm ? mm_ra_l : l1;
    assign p_rq_valid[2] = is_mm ? mm_rb_v : v2;
    assign p_rq_line[52 +: 26] = is_mm ? mm_rb_l : l2;
    assign p_rq_valid[3] = v3;
    assign p_rq_line[78 +: 26] = l3;
    assign p_rq_valid[4] = v4;
    assign p_rq_line[104 +: 26] = l4;
    assign p_rq_we[4:1] = 4'b0;
    assign p_rq_wdata[5*LB*8-1:LB*8] = 0;
    assign p_rq_wstrb[5*LB-1:LB] = 0;

    // zapisovac (port 0)
    reg  e_w_start, e_w_valid, w_flush;
    reg  [VL*64-1:0] e_w_data;
    wire w_busy;
    bittf_wstream #(.VL(VL), .LB(LB)) wr (
        .clk(clk), .rst(rst),
        .start(is_mm ? mm_w_start : e_w_start),
        .base(is_mm ? mm_w_base : obase(0)), .lw(olw(0)),
        .vec(is_mm ? mm_w_vec : (is_red ? 1'b0 : vec)),
        .cnt(is_mm ? mm_w_cnt : (is_red ? {ir[4], ir[3], ir[2], 32'd1} : cntv)),
        .str(is_mm ? mm_w_str : (is_red ? {ostr(0, 3), ostr(0, 2), ostr(0, 1), 64'sd0} : ostrv(0))),
        .flush(w_flush), .i_valid(is_mm ? mm_w_valid : e_w_valid), .i_ready(w_ready),
        .i_data(is_mm ? mm_w_data : e_w_data),
        .wq_valid(w_rq_valid), .wq_ready(p_rq_ready[0]), .wq_line(w_rq_line), .wq_data(w_rq_wdata),
        .wq_strb(w_rq_wstrb), .wack(p_wack[0] && use_w), .busy(w_busy));

    // ------------------------------------------------------------ smycky taktu (indexy pro masku, RNG, konec radku)
    reg [31:0] cnt [0:3];
    reg [31:0] idx [0:3];
    reg [31:0] flat;
    wire [31:0] rem0 = cnt[0] - idx[0];
    wire [6:0] blen = (!vec) ? 7'd1 : (rem0 >= VL) ? VL[6:0] : rem0[6:0];
    reg [2:0] adv;
    always @(*) begin
        if (idx[0] + blen < cnt[0]) adv = 3'd0;
        else if (idx[1] + 1 < cnt[1]) adv = 3'd1;
        else if (idx[2] + 1 < cnt[2]) adv = 3'd2;
        else if (idx[3] + 1 < cnt[3]) adv = 3'd3;
        else adv = 3'd4;
    end
    task loop_adv;
        integer kk;
        begin
            for (kk = 0; kk < 4; kk = kk + 1) begin
                if (kk < adv) idx[kk] <= 32'd0;
                else if (kk == adv) idx[kk] <= idx[kk] + ((kk == 0) ? {25'd0, blen} : 32'd1);
            end
            flat <= flat + {25'd0, blen};
        end
    endtask

    // ------------------------------------------------------------ drahy EW
    reg [31:0] key;
    wire [VL*64-1:0] ly;
    genvar g;
    generate
        for (g = 0; g < VL; g = g + 1) begin : ln
            wire [31:0] rnd = f_fmix32((flat + g) * 32'h9E3779B1 + key);
            bittf_lane lane (
                .fn(fn), .tc(tc), .clamp_en(fl[3] | fl[4]), .clamp_sym(fl[4]), .cbits(imm3[6:0]),
                .a(od1[64 * g +: 64]), .b(od2[64 * g +: 64]), .c(od3[64 * g +: 64]), .r1(vR1), .r2(vR2), .imm(imm),
                .i0(idx[0] + g), .i1(idx[1]), .rnd(rnd), .y(ly[64 * g +: 64]));
        end
    endgenerate
    reg [VL*64-1:0] ew_y;
    reg ew_fit;
    always @(*) begin
        ew_fit = 1'b1;
        for (l = 0; l < VL; l = l + 1) begin
            ew_y[64 * l +: 64] = (fl[5] && idx[0] + l > idx[1]) ? vR2 : ly[64 * l +: 64];
            if (l < blen && !fits(ew_y[64 * l +: 64], olw(0), ouns(0))) ew_fit = 1'b0;
        end
    end

    // skalarni ALU (deleni, gather, nesouvisle operandy)
    reg alu_start;
    wire alu_done;
    wire [63:0] alu_y;
    bittf_alu alu (
        .clk(clk), .rst(rst), .start(alu_start), .fn(fn), .tc(tc), .clamp_en(fl[3] | fl[4]), .clamp_sym(fl[4]),
        .cbits(imm3[6:0]), .a(od1[63:0]), .b(od2[63:0]), .c(od3[63:0]), .r1(vR1), .r2(vR2), .imm(imm),
        .i0(idx[0]), .i1(idx[1]), .rnd(ln[0].rnd), .done(alu_done), .y(alu_y));
    wire signed [63:0] alu_v = (fl[5] && idx[0] > idx[1]) ? vR2 : $signed(alu_y);

    // ------------------------------------------------------------ RED
    reg signed [63:0] acc;
    reg signed [63:0] bv;            // kombinace drah taktu
    reg signed [63:0] lv, lc, lt;
    always @(*) begin
        bv = (fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
        for (l = 0; l < VL; l = l + 1) begin
            lc = $signed(od3[64 * l +: 64]);
            case (tc)
                TC_ADD: lt = lc;
                TC_SUB: lt = -lc;
                TC_ADDHALF: lt = lc >>> 1;
                TC_SUBHALF: lt = -(lc >>> 1);
                default: lt = 64'sd0;
            endcase
            case (rpre)
                RP_AA:  lv = $signed(od1[64 * l +: 64]) * $signed(od1[64 * l +: 64]);
                RP_AB:  lv = $signed(od1[64 * l +: 64]) * $signed(od2[64 * l +: 64]);
                RP_ABC: lv = $signed(od1[64 * l +: 64]) * $signed(od2[64 * l +: 64]) * lc;
                RP_SH:  lv = simm + vR1 + lt;
                default: lv = $signed(od1[64 * l +: 64]);
            endcase
            if (l < blen) begin
                case (fn)
                    R_SUM: bv = bv + lv;
                    R_MAX: bv = lv > bv ? lv : bv;
                    default: bv = (lv[63] ? -lv : lv) > bv ? (lv[63] ? -lv : lv) : bv;
                endcase
            end
        end
    end
    reg signed [63:0] acc_n;
    always @(*) begin
        case (fn)
            R_SUM: acc_n = acc + bv;
            R_MAX: acc_n = bv > acc ? bv : acc;
            default: acc_n = bv > acc ? bv : acc;
        endcase
    end

    // pripravenost operandu taktu
    wire ops_ok = (!need_a || ov[1]) && (!need_b || ov[2]) && (!need_c || ov[3]);
    wire wslot = !e_w_valid || w_ready;      // takt k zapisu muze nastoupit tento cyklus
    wire ew_fire = st == S_EW && vec && ops_ok && wslot;
    wire red_row = adv != 0;
    wire red_fire = st == S_RED && ops_ok && (!red_row || fl[11] || wslot);
    reg  sc_pop;                       // skalarni EW: vyzvednuti po dokonceni ALU
    wire pop = ew_fire || red_fire || sc_pop;
    assign oready[1] = pop && need_a;
    assign oready[2] = pop && need_b;
    assign oready[3] = pop && need_c;
    reg last_done;                     // vsechny takty zpracovany

    // ------------------------------------------------------------ skalarni jednotka
    reg signed [63:0] sres;
    reg swr;
    always @(*) begin
        swr = 1'b1;
        case (fn)
            S_LI:     sres = simm + (simm2 <<< 32);
            S_ADD:    sres = vR1 + vR2 + simm;
            S_SUB:    sres = vR1 - vR2 + simm;
            S_MAX:    sres = vR1 > vR2 ? vR1 : vR2;
            S_MIN:    sres = vR1 < vR2 ? vR1 : vR2;
            S_BITLEN: sres = $signed({57'd0, f_bitlen(vR1)}) + simm;
            S_SEL:    sres = vR3 != 0 ? vR1 : vR2;
            S_SHRA:   sres = vR1 >>> imm[5:0];
            S_NE:     sres = {63'd0, vR1 != vR2};
            S_CSRR:   sres = {32'd0, imm[1:0] == 0 ? csr_seed : imm[1:0] == 1 ? csr_ctr :
                              imm[1:0] == 2 ? {24'd0, err} : csr_cyc};
            default:  begin sres = 64'sd0; swr = 1'b0; end
        endcase
    end

    function signed [63:0] ext(input [63:0] v, input [1:0] lw_, input uns);
        case (lw_)
            2'd0: ext = uns ? {56'd0, v[7:0]} : {{56{v[7]}}, v[7:0]};
            2'd1: ext = uns ? {48'd0, v[15:0]} : {{48{v[15]}}, v[15:0]};
            2'd2: ext = uns ? {32'd0, v[31:0]} : {{32{v[31]}}, v[31:0]};
            default: ext = v;
        endcase
    endfunction
    function fits(input signed [63:0] v, input [1:0] lw_, input uns);
        case (lw_)
            2'd0: fits = uns ? v[63:8] == 0 : (v[63:7] == 0 || &v[63:7]);
            2'd1: fits = uns ? v[63:16] == 0 : (v[63:15] == 0 || &v[63:15]);
            2'd2: fits = uns ? v[63:32] == 0 : (v[63:31] == 0 || &v[63:31]);
            default: fits = 1'b1;
        endcase
    endfunction

    // ------------------------------------------------------------ hlavni automat
    reg [1:0] fcnt, frsp;
    wire [31:0] iaddr = base + pc * 192;
    wire [31:0] saddr = obase(0);
    wire [LB*8-1:0] sline = rs0 >> {saddr[LBB-1:0], 3'b000};
    reg [7:0] wbytes;
    reg wait_ack;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE;
            done <= 1'b0;
            err <= 8'd0;
            c_rq_valid <= 1'b0;
            dbg_instr_done <= 1'b0;
            csr_seed <= 0; csr_ctr <= 0; csr_cyc <= 0;
            s_start <= 1'b0; mm_start <= 1'b0; e_w_start <= 1'b0; e_w_valid <= 1'b0; w_flush <= 1'b0;
            alu_start <= 1'b0; sc_pop <= 1'b0;
            for (k = 0; k < 64; k = k + 1) R[k] <= 64'd0;
        end else begin
            done <= 1'b0;
            dbg_instr_done <= 1'b0;
            s_start <= 1'b0;
            mm_start <= 1'b0;
            e_w_start <= 1'b0;
            alu_start <= 1'b0;
            sc_pop <= 1'b0;
            if (st != S_IDLE) csr_cyc <= csr_cyc + 32'd1;
            if (c_rq_valid && p_rq_ready[0] && !use_w) c_rq_valid <= 1'b0;
            if (e_w_valid && w_ready) e_w_valid <= 1'b0;
            if (csr_we && st == S_IDLE) begin
                case (csr_addr)
                    2'd0: csr_seed <= csr_wdata;
                    2'd1: csr_ctr <= csr_wdata;
                    2'd2: err <= csr_wdata[7:0];
                    default: csr_cyc <= csr_wdata;
                endcase
            end
            case (st)
                S_IDLE:
                    if (start) begin
                        base <= prog_base;
                        pc <= 0;
                        err <= 0;
                        fcnt <= 0;
                        frsp <= 0;
                        st <= S_FETCH;
                    end
                S_FETCH: begin
                    if (p_rs_valid[0]) begin
                        for (k = 0; k < 16; k = k + 1)
                            ir[16 * frsp + k] <= rs0[32 * k +: 32];
                        frsp <= frsp + 1;
                        if (frsp == 2) st <= S_DEC;
                    end
                    if ((!c_rq_valid || p_rq_ready[0]) && fcnt < 3) begin
                        c_rq_valid <= 1'b1;
                        c_rq_we <= 1'b0;
                        c_rq_line <= iaddr[31:LBB] + fcnt;
                        fcnt <= fcnt + 1;
                    end
                end
                S_DEC: begin
                    dbg_pc <= pc;
                    case (op)
                        OP_NOP: st <= S_END;
                        OP_HALT: st <= S_HALT;
                        OP_SCALAR: st <= S_SCAL;
                        OP_BRANCH: begin
                            if ((fn == B_BEQZ && vR1 == 0) || (fn == B_BNEZ && vR1 != 0) || fn == B_JMP)
                                pc <= imm - 32'd1;
                            st <= S_END;
                        end
                        OP_EW, OP_RED: begin
                            vec <= vec_ok;
                            for (k = 0; k < 4; k = k + 1) begin
                                cnt[k] <= ir[1 + k];
                                idx[k] <= 0;
                            end
                            flat <= 0;
                            acc <= (is_red && fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                            if (uses_rng) begin
                                key <= f_fmix32(csr_seed ^ f_fmix32(csr_ctr));
                                csr_ctr <= csr_ctr + 32'd1;
                            end
                            last_done <= 1'b0;
                            st <= S_SETUP;
                        end
                        OP_MM: begin
                            mm_start <= 1'b1;
                            st <= S_MM;
                        end
                        default: begin err <= ERR_OP; st <= S_ERR; end
                    endcase
                end
                S_SETUP: begin
                    // vec je nastaveno: spustit proudy a zapisovac
                    s_start <= 1'b1;
                    if (!(is_red && fl[11])) e_w_start <= 1'b1;
                    st <= is_ew ? S_EW : S_RED;
                end
                // --- skalarni instrukce
                S_SCAL: begin
                    if (fn == S_LD) begin
                        c_rq_valid <= 1'b1;
                        c_rq_we <= 1'b0;
                        c_rq_line <= saddr[31:LBB];
                        st <= S_SLD;
                    end else if (fn == S_ST) begin
                        if (!fits(vR1, olw(0), ouns(0))) begin
                            err <= ERR_FIT; st <= S_ERR;
                        end else begin
                            c_rq_valid <= 1'b1;
                            c_rq_we <= 1'b1;
                            c_rq_line <= saddr[31:LBB];
                            c_rq_wdata <= {{(LB*8-64){1'b0}}, vR1} << {saddr[LBB-1:0], 3'b000};
                            c_rq_wstrb <= {{(LB-8){1'b0}}, ((8'd1 << (8'd1 << olw(0))) - 8'd1)} << saddr[LBB-1:0];
                            st <= S_SST;
                        end
                    end else begin
                        if (fn == S_CSRW) begin
                            case (imm[1:0])
                                2'd0: csr_seed <= vR1[31:0];
                                2'd1: csr_ctr <= vR1[31:0];
                                2'd2: err <= vR1[7:0];
                                default: csr_cyc <= vR1[31:0];
                            endcase
                            st <= S_END;
                        end else if (swr) begin
                            if (rd != 0) R[rd] <= sres;
                            st <= S_END;
                        end else begin
                            err <= ERR_OP; st <= S_ERR;
                        end
                    end
                end
                S_SLD:
                    if (p_rs_valid[0]) begin
                        if (rd != 0) R[rd] <= ext(sline[63:0], olw(0), ouns(0));
                        st <= S_END;
                    end
                S_SST:
                    if (p_wack[0]) st <= S_END;
                // --- EW
                S_EW: begin
                    if (ew_fire) begin
                        if (!ew_fit) begin
                            err <= ERR_FIT; st <= S_ERR;
                        end else begin
                            e_w_data <= ew_y;
                            e_w_valid <= 1'b1;
                            loop_adv;
                            if (adv == 4) begin st <= S_DRAIN; w_flush <= 1'b1; end
                        end
                    end else if (!vec && ops_ok && wslot && !sc_pop && !last_done) begin
                        if (fl[5] && idx[0] > idx[1]) begin
                            // maskovany prvek: bez ALU
                            e_w_data <= {{(VL*64-64){1'b0}}, vR2};
                            e_w_valid <= 1'b1;
                            sc_pop <= 1'b1;
                            loop_adv;
                            if (adv == 4) begin st <= S_DRAIN; w_flush <= 1'b1; end
                        end else begin
                            alu_start <= 1'b1;
                            st <= S_ALUW;
                        end
                    end
                end
                S_ALUW:
                    if (alu_done) begin
                        if (!fits(alu_y, olw(0), ouns(0))) begin
                            err <= ERR_FIT; st <= S_ERR;
                        end else begin
                            e_w_data <= {{(VL*64-64){1'b0}}, alu_y};
                            e_w_valid <= 1'b1;
                            sc_pop <= 1'b1;
                            loop_adv;
                            if (adv == 4) begin st <= S_DRAIN; w_flush <= 1'b1; end
                            else st <= S_EW;
                        end
                    end
                // --- RED
                S_RED:
                    if (red_fire) begin
                        if (red_row) begin
                            if (fl[11]) begin
                                if (rd != 0) R[rd] <= acc_n;
                            end else if (!fits(acc_n, olw(0), ouns(0))) begin
                                err <= ERR_FIT; st <= S_ERR;
                            end else begin
                                e_w_data <= {{(VL*64-64){1'b0}}, acc_n};
                                e_w_valid <= 1'b1;
                            end
                            acc <= (fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                        end else
                            acc <= acc_n;
                        loop_adv;
                        if (adv == 4) begin st <= S_DRAIN; w_flush <= 1'b1; end
                    end
                // --- MM
                S_MM:
                    if (mm_done) begin
                        if (mm_err) begin err <= ERR_FIT; st <= S_ERR; end
                        else begin st <= S_DRAIN; w_flush <= 1'b1; end
                    end
                S_DRAIN:
                    if (!e_w_valid && !w_busy && sbusy == 0) begin
                        w_flush <= 1'b0;
                        st <= S_END;
                    end
                S_END: begin
                    dbg_instr_done <= 1'b1;
                    pc <= pc + 1;
                    fcnt <= 0;
                    frsp <= 0;
                    st <= S_FETCH;
                end
                S_HALT: begin
                    done <= 1'b1;
                    dbg_instr_done <= 1'b1;
                    st <= S_IDLE;
                end
                S_ERR: begin
                    done <= 1'b1;
                    w_flush <= 1'b0;
                    st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
            // skalarni EW ceka v S_EW na dalsi prvek po sc_pop
            if (st == S_ALUW && alu_done && adv != 4 && fits(alu_y, olw(0), ouns(0))) st <= S_EW;
        end
    end
endmodule
