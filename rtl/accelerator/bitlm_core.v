`timescale 1ns/1ps
// =============================================================================
// bitlm_core - HW trenink a inference 1-bitoveho jazykoveho modelu (BitLM-nano)
//
// Bitove presna implementace referencniho modelu sw/bitlm/golden/bitlm_ref.c
// (varianta hparams.flags = 0: bez centrovani vah a bez biasu logitu).
// Cely treninkovy krok bezi v HW: vyber vzorku z korpusu (RNG), dopredny
// pruchod, softmax, zpetny pruchod a update stinovych vah i embeddingu.
// Host pouze nahraje korpus, nastavi hyperparametry a spusti N kroku.
//
// Datove toky:
//   W1 (H x D) a W2 (V x H): stinove vahy int16 v BRAM, slovo = L lane.
//   1-bit vaha = znamenko stinove vahy -> dopredny/zpetny pruchod jen +-.
//   Kazdy pruchod vahami je proud: 1 slovo / takt, latence cteni 2 takty.
//
// MMIO (bajtove adresy, 32-bit slova; viz docs/bitlm_poc.md):
//   0x000000 registry, 0x010000 generovany text, 0x400000 korpus (4 MiB),
//   0x800000 W1 (4 MiB), 0xC00000 W2 (1 MiB), 0xD00000 embedding (64 KiB).
// Zapis je okamzity; cteni vraci mmio_rvalid po 1-3 taktech.
// Pameti jsou z MMIO pristupne jen v klidu (busy = 0).
// =============================================================================
module bitlm_core #(
    parameter LOG2_N = 3,   // kontext 8
    parameter LOG2_E = 5,   // embedding 32
    parameter LOG2_H = 9,   // skryta vrstva 512
    parameter LOG2_L = 6,   // 64 lane (paralelnich +-/MAC) pro W1/W2
    parameter LOG2_CORPUS = 22, // korpus az 4 Mi tokenu
    parameter MULTISCALE = 0    // 1: kontext 128 -> 16 slotu (8 x 1 znak + prumery oken), viz bitlm_ref.h
) (
    input  wire        clk,
    input  wire        rst,

    input  wire        mmio_we,
    input  wire        mmio_re,
    input  wire [23:0] mmio_addr,
    input  wire [31:0] mmio_wdata,
    output reg  [31:0] mmio_rdata,
    output reg         mmio_rvalid,

    output wire        busy
);
    // ------------------------------------------------------------------ rozmery
    localparam LOG2_V = 5;
    localparam V = 1 << LOG2_V;
    localparam N = 1 << LOG2_N;
    localparam E = 1 << LOG2_E;
    localparam LOG2_SLOTS = MULTISCALE ? 4 : LOG2_N;
    localparam LOG2_D = LOG2_SLOTS + LOG2_E;
    localparam D = 1 << LOG2_D;
    localparam H = 1 << LOG2_H;
    localparam L = 1 << LOG2_L;
    localparam D_BLK_LOG2 = LOG2_D - LOG2_L;       // slov na radek W1
    localparam H_BLK_LOG2 = LOG2_H - LOG2_L;       // slov na radek W2
    localparam W1_WORDS = H << D_BLK_LOG2;
    localparam W2_WORDS = V << H_BLK_LOG2;
    localparam W1_AW = LOG2_H + D_BLK_LOG2;
    localparam W2_AW = LOG2_V + H_BLK_LOG2;
    localparam LOG2_GEN = 12;                      // 4 Ki tokenu

    localparam [31:0] ID_VALUE = 32'h424C_4D31;    // "BLM1"
    localparam [7:0]  HW_VERSION = 8'd2;   // v2: 32-bit vyber pozice, vetsi okna MMIO
    localparam [3:0]  P_N = LOG2_N, P_E = LOG2_E, P_V = LOG2_V, P_H = LOG2_H, P_L = LOG2_L;
    localparam [3:0]  P_MS = MULTISCALE ? 4'd1 : 4'd0;

    // kontrola parametru (simulace i syntéza hlasi chybu)
    initial begin
        if (LOG2_L > LOG2_D || LOG2_L > LOG2_H) $display("bitlm_core: LOG2_L musi byt <= LOG2_D, LOG2_H");
        if (LOG2_E > LOG2_L) $display("bitlm_core: LOG2_E musi byt <= LOG2_L (init buffer)");
        if (MULTISCALE && LOG2_N != 7) $display("bitlm_core: MULTISCALE vyzaduje LOG2_N = 7 (kontext 128)");
        if ((H * D) > 1048576 || (V * H) > 262144 || (V * E) > 16384 || LOG2_CORPUS > 22 || LOG2_N > 7)
            $display("bitlm_core: rozmery presahuji MMIO okna nebo citace (viz docs/bitlm_poc.md)");
    end

    `include "bitlm_exp_lut.vh"

    // ------------------------------------------------------- pomocne funkce
    // zaokrouhlujici aritmeticky posun: (v + 2^(s-1)) >>> s, s = 0 -> v
    function signed [39:0] rsr40;
        input signed [39:0] v;
        input [5:0] s;
        begin
            if (s == 6'd0) rsr40 = v;
            else rsr40 = (v + (40'sd1 <<< (s - 6'd1))) >>> s;
        end
    endfunction

    // rozlozeni slotu kontextu (MULTISCALE): okna 32,32,16,16,8,8,4,4 a 8 x 1 znak
    function [3:0] slot_of;   // slot pozice n (jen MULTISCALE)
        input [6:0] n;
        begin
            if (n < 7'd32) slot_of = 4'd0;
            else if (n < 7'd64) slot_of = 4'd1;
            else if (n < 7'd80) slot_of = 4'd2;
            else if (n < 7'd96) slot_of = 4'd3;
            else if (n < 7'd104) slot_of = 4'd4;
            else if (n < 7'd112) slot_of = 4'd5;
            else if (n < 7'd116) slot_of = 4'd6;
            else if (n < 7'd120) slot_of = 4'd7;
            else slot_of = 4'd8 + {1'b0, n[2:0]};
        end
    endfunction
    function [2:0] slot_lg;   // log2 velikosti okna slotu
        input [3:0] sl;
        begin
            case (sl)
                4'd0, 4'd1: slot_lg = 3'd5;
                4'd2, 4'd3: slot_lg = 3'd4;
                4'd4, 4'd5: slot_lg = 3'd3;
                4'd6, 4'd7: slot_lg = 3'd2;
                default:    slot_lg = 3'd0;
            endcase
        end
    endfunction
    function slot_last;       // posledni pozice okna
        input [6:0] n;
        begin
            slot_last = (n == 7'd31) || (n == 7'd63) || (n == 7'd79) || (n == 7'd95) ||
                        (n == 7'd103) || (n == 7'd111) || (n == 7'd115) || (n >= 7'd119);
        end
    endfunction

    // oriznuti do <-c, c>
    function signed [15:0] clipw;
        input signed [39:0] v;
        input [15:0] c;
        reg signed [39:0] cs;
        begin
            cs = $signed({24'd0, c});
            if (v > cs) clipw = cs[15:0];
            else if (v < -cs) clipw = -cs[15:0];
            else clipw = v[15:0];
        end
    endfunction

    // x okna = clamp(sum >>> (8 + lg), -127, 127)
    function signed [7:0] win2x;
        input signed [23:0] sm;
        input [2:0] lg;
        reg signed [23:0] t;
        begin
            t = sm >>> (5'd8 + {2'd0, lg});
            win2x = (t < -24'sd127) ? -8'sd127 : ((t > 24'sd127) ? 8'sd127 : t[7:0]);
        end
    endfunction

    // x = clamp(emb >>> 8, -127, 127)
    function signed [7:0] emb2x;
        input signed [15:0] e;
        reg signed [15:0] t;
        begin
            t = e >>> 8;
            emb2x = (t < -16'sd127) ? -8'sd127 : t[7:0];
        end
    endfunction

    function [31:0] xorshift32;
        input [31:0] s;
        reg [31:0] x;
        begin
            x = s ^ (s << 13);
            x = x ^ (x >> 17);
            xorshift32 = x ^ (x << 5);
        end
    endfunction

    // nejmensi posun s >= smin, aby (hm >> s) <= 127
    function [4:0] dyn_shift;
        input [23:0] hm;
        input [4:0] smin;
        integer k;
        reg found;
        begin
            dyn_shift = 5'd31;
            found = 1'b0;
            for (k = 0; k < 32; k = k + 1)
                if (!found && k >= smin && (hm >> k) <= 24'd127) begin
                    dyn_shift = k[4:0];
                    found = 1'b1;
                end
        end
    endfunction

    // ------------------------------------------------------------- registry
    reg [31:0] seed_r, nsteps_r, gen_len_r;
    reg [LOG2_CORPUS:0] corpus_len_r;              // az 131072
    reg [4:0]  hp_s1, hp_s2, hp_g1, hp_g2, hp_lr1, hp_lr2, hp_lre, hp_lrb;
    reg [15:0] hp_clip;
    reg [7:0]  hp_flags;                           // HW v1: ignoruje (flags = 0)
    reg [31:0] steps_done, correct_cnt;
    reg [63:0] sum_pt;
    reg [63:0] cycles;
    reg [31:0] rng;
    reg        err_r, done_r;
    reg [4:0]  ctx [0:N-1];

    // ---------------------------------------------------------- vektory dat
    reg signed [7:0]  xv  [0:D-1];
    reg signed [23:0] esumv [0:E-1];       // soucty embeddingu v okne (dopredny pruchod)
    reg signed [23:0] hv  [0:H-1];
    reg        [6:0]  av  [0:H-1];
    reg signed [31:0] dav [0:H-1];
    reg signed [31:0] dhv [0:H-1];
    reg signed [31:0] dxv [0:D-1];
    reg signed [23:0] zv  [0:V-1];
    reg        [16:0] ev  [0:V-1];
    reg        [16:0] pv  [0:V-1];
    reg signed [17:0] gzv [0:V-1];

    // ------------------------------------------------------------- pameti
    // W1
    reg [16*L-1:0] w1_mem [0:W1_WORDS-1];
    reg [W1_AW-1:0] w1_raddr, w1_waddr;
    reg [16*L-1:0] w1_rdata, w1_wdata;
    reg [L-1:0]    w1_we;
    // W2
    reg [16*L-1:0] w2_mem [0:W2_WORDS-1];
    reg [W2_AW-1:0] w2_raddr, w2_waddr;
    reg [16*L-1:0] w2_rdata, w2_wdata;
    reg [L-1:0]    w2_we;
    // embedding: slovo = jeden znak (E lane)
    reg [16*E-1:0] emb_mem [0:V-1];
    reg [LOG2_V-1:0] emb_raddr, emb_waddr;
    reg [16*E-1:0] emb_rdata, emb_wdata;
    reg [E-1:0]    emb_we;
    // korpus: 4 tokeny ve slove
    reg [31:0] corp_mem [0:(1 << (LOG2_CORPUS - 2)) - 1];
    reg [LOG2_CORPUS-3:0] corp_raddr, corp_waddr;
    reg [31:0] corp_rdata, corp_wdata;
    reg        corp_we;
    // generovany text: 4 tokeny ve slove, zapis po bajtech
    reg [31:0] gen_mem [0:(1 << (LOG2_GEN - 2)) - 1];
    reg [LOG2_GEN-3:0] gen_raddr, gen_waddr;
    reg [31:0] gen_rdata, gen_wdata;
    reg [3:0]  gen_we;

    integer li;
    always @(posedge clk) begin
        for (li = 0; li < L; li = li + 1)
            if (w1_we[li]) w1_mem[w1_waddr][li*16 +: 16] <= w1_wdata[li*16 +: 16];
        w1_rdata <= w1_mem[w1_raddr];
    end
    always @(posedge clk) begin
        for (li = 0; li < L; li = li + 1)
            if (w2_we[li]) w2_mem[w2_waddr][li*16 +: 16] <= w2_wdata[li*16 +: 16];
        w2_rdata <= w2_mem[w2_raddr];
    end
    always @(posedge clk) begin
        for (li = 0; li < E; li = li + 1)
            if (emb_we[li]) emb_mem[emb_waddr][li*16 +: 16] <= emb_wdata[li*16 +: 16];
        emb_rdata <= emb_mem[emb_raddr];
    end
    always @(posedge clk) begin
        if (corp_we) corp_mem[corp_waddr] <= corp_wdata;
        corp_rdata <= corp_mem[corp_raddr];
    end
    always @(posedge clk) begin
        for (li = 0; li < 4; li = li + 1)
            if (gen_we[li]) gen_mem[gen_waddr][li*8 +: 8] <= gen_wdata[li*8 +: 8];
        gen_rdata <= gen_mem[gen_raddr];
    end

    // ---------------------------------------------------------------- stavy
    localparam [5:0]
        S_IDLE    = 6'd0,
        S_INIT    = 6'd1,
        S_INIT_WR = 6'd2,
        S_T_POS   = 6'd3,
        S_T_FETCH = 6'd4,
        S_F_EMB   = 6'd5,
        S_F_W1    = 6'd6,
        S_F_ACT   = 6'd7,
        S_F_W2    = 6'd8,
        S_F_EXP   = 6'd9,
        S_F_DIV   = 6'd10,
        S_F_PROB  = 6'd11,
        S_B_W2    = 6'd12,
        S_B_DH    = 6'd13,
        S_B_W1    = 6'd14,
        S_B_EMB   = 6'd15,
        S_G_SAMP  = 6'd16,
        S_G_EMIT  = 6'd17,
        S_DONE    = 6'd18;

    reg [5:0] state;
    reg       mode_gen;                // 0 trenink, 1 generovani
    assign busy = (state != S_IDLE);

    // citace pruchodu
    reg [31:0] steps_left, gen_left;
    reg [LOG2_GEN-1:0] gen_ptr;
    reg [LOG2_CORPUS:0] pos;
    reg [4:0]  target;
    reg [15:0] i_row;                  // radek (j nebo i)
    reg [15:0] i_blk;                  // slovo v radku
    reg        issuing;                // 1 = vydavaji se adresy, 0 = doznivani
    reg [2:0]  drain;
    reg [7:0]  sub;                    // pomocny citac (EMB update, deleni)

    // pipeline cteni: p1 = adresa v registru, p2 = data v *_rdata
    reg        p1_v, p2_v;
    reg [15:0] p1_row, p2_row, p1_blk, p2_blk;
    reg signed [31:0] p1_dh, p2_dh;

    // init
    reg [1:0]  init_sel;               // 0 W1, 1 W2, 2 EMB
    reg [15:0] init_addr;
    reg [7:0]  init_lane;
    reg [16*L-1:0] ibuf;

    // forward
    reg signed [31:0] acc;
    reg [23:0] hmax;
    reg [4:0]  s1dyn;
    reg signed [23:0] zmax;
    reg [4:0]  argmax;
    reg [22:0] esum;
    reg [32:0] div_rem;
    reg [32:0] div_q;
    reg [16:0] rcp;
    reg [16:0] cum;
    reg        found;
    reg [4:0]  tok;
    reg [15:0] u16;

    // -------------------------------------------- kombinacni vypocty pruchodu
    // (pouzivaji data v p2 stupni)
    reg signed [31:0] fw1_part, fw2_part;
    reg [16*L-1:0] bw2_new, bw1_new;
    reg [16*E-1:0] emb_new;
    reg signed [15:0] wl;
    reg signed [39:0] prod, upd;
    integer cl, ck;
    always @* begin
        fw1_part = 32'sd0;
        fw2_part = 32'sd0;
        bw2_new = {16*L{1'b0}};
        bw1_new = {16*L{1'b0}};
        emb_new = {16*E{1'b0}};
        wl = 16'sd0;
        prod = 40'sd0;
        upd = 40'sd0;
        for (cl = 0; cl < L; cl = cl + 1) begin
            // dopredny W1: +-x
            wl = $signed(w1_rdata[cl*16 +: 16]);
            ck = (p2_blk << LOG2_L) + cl;
            if (ck < D) begin
                if (!wl[15]) fw1_part = fw1_part + xv[ck];
                else         fw1_part = fw1_part - xv[ck];
                // zpetny W1: w -= rsr(dh * x, lr1)
                prod = p2_dh * xv[ck];
                upd = rsr40(prod, hp_lr1);
                bw1_new[cl*16 +: 16] = clipw($signed({{24{wl[15]}}, wl}) - upd, hp_clip);
            end
            // dopredny W2: +-a
            wl = $signed(w2_rdata[cl*16 +: 16]);
            ck = (p2_blk << LOG2_L) + cl;
            if (ck < H) begin
                if (!wl[15]) fw2_part = fw2_part + $signed({25'd0, av[ck]});
                else         fw2_part = fw2_part - $signed({25'd0, av[ck]});
                // zpetny W2: w -= rsr(gz * a, lr2)
                prod = gzv[p2_row[LOG2_V-1:0]] * $signed({33'd0, av[ck]});
                upd = rsr40(prod, hp_lr2);
                bw2_new[cl*16 +: 16] = clipw($signed({{24{wl[15]}}, wl}) - upd, hp_clip);
            end
        end
        // embedding update pro pozici sub (n): emb -= rsr(dx[slot*E+e], lre + log2 okna)
        for (cl = 0; cl < E; cl = cl + 1) begin
            wl = $signed(emb_rdata[cl*16 +: 16]);
            ck = ({8'd0, bemb_slot} << LOG2_E) + cl;
            upd = rsr40($signed({{8{dxv[ck][31]}}, dxv[ck]}), {1'b0, hp_lre} + {3'd0, bemb_lg});
            emb_new[cl*16 +: 16] = clipw($signed({{24{wl[15]}}, wl}) - upd, 16'd32767);
        end
    end

    // pocet moznych pozic vzorku: len - N
    localparam [LOG2_CORPUS:0] N_W = N;
    wire [LOG2_CORPUS:0] len_minus_n = corpus_len_r - N_W;

    // index tokenu pri cteni kontextu z korpusu
    wire [LOG2_CORPUS:0] fetch_idx = pos - N + {{(LOG2_CORPUS-7){1'b0}}, sub};

    // slot a log2 okna pro pozici sub (zpetny pruchod embeddingu) a pro p2_row (dopredny)
    wire [7:0] bemb_slot = MULTISCALE ? {4'd0, slot_of(sub[6:0])} : sub;
    wire [2:0] bemb_lg   = MULTISCALE ? slot_lg(slot_of(sub[6:0])) : 3'd0;
    wire [7:0] femb_slot = MULTISCALE ? {4'd0, slot_of(p2_row[6:0])} : p2_row[7:0];
    wire [2:0] femb_lg   = MULTISCALE ? slot_lg(slot_of(p2_row[6:0])) : 3'd0;
    wire       femb_last = MULTISCALE ? slot_last(p2_row[6:0]) : 1'b1;

    // dh vybraneho radku pro W1 zpetny pruchod
    wire signed [31:0] dh_sel = dhv[i_row[LOG2_H-1:0]];

    // pocatecni hodnoty z RNG (stejne jako bl_init)
    wire [31:0] rng_nx = xorshift32(rng);

    // nahodna pozice vzorku: N + (rng * (len - N)) >> 32 (vsechny pozice dosazitelne)
    wire [32 + LOG2_CORPUS:0] pos_prod = {{(LOG2_CORPUS + 1){1'b0}}, rng_nx} * {32'd0, len_minus_n};
    wire [LOG2_CORPUS:0] pos_next = N_W + pos_prod[32 + LOG2_CORPUS:32];
    wire signed [15:0] init_w = $signed({7'd0, rng_nx[24:16]}) - 16'sd256;
    wire signed [15:0] init_e = ($signed({10'd0, rng_nx[29:24]}) - 16'sd32) <<< 8;

    // ------------------------------------------------------------- MMIO cteni
    reg [2:0]  mr_cnt;
    reg [2:0]  mr_kind;                // 1 W1, 2 W2, 3 EMB, 4 korpus, 5 gen
    reg [7:0]  mr_lane;

    function [31:0] sx16;
        input [15:0] v;
        sx16 = {{16{v[15]}}, v};
    endfunction

    reg [31:0] reg_rdata;
    integer cn;
    always @* begin
        reg_rdata = 32'd0;
        case (mmio_addr[7:0])
            8'h00: reg_rdata = ID_VALUE;
            8'h04: reg_rdata = {HW_VERSION, P_L, P_H, P_MS, P_V, P_E, P_N};
            8'h0C: reg_rdata = {16'd0, 2'd0, state, 5'd0, err_r, done_r, busy};
            8'h10: reg_rdata = seed_r;
            8'h14: reg_rdata = nsteps_r;
            8'h18: reg_rdata = {{(31 - LOG2_CORPUS){1'b0}}, corpus_len_r};
            8'h1C: reg_rdata = gen_len_r;
            8'h20: reg_rdata = {3'd0, hp_g2, 3'd0, hp_g1, 3'd0, hp_s2, 3'd0, hp_s1};
            8'h24: reg_rdata = {3'd0, hp_lrb, 3'd0, hp_lre, 3'd0, hp_lr2, 3'd0, hp_lr1};
            8'h28: reg_rdata = {8'd0, hp_flags, hp_clip};
            8'h2C: reg_rdata = steps_done;
            8'h30: reg_rdata = correct_cnt;
            8'h34: reg_rdata = sum_pt[31:0];
            8'h38: reg_rdata = sum_pt[63:32];
            8'h3C: reg_rdata = rng;
            8'h40: reg_rdata = cycles[31:0];
            8'h44: reg_rdata = cycles[63:32];
            default: begin
                // 0x80.. kontext: 4 tokeny na slovo
                if (mmio_addr[7] && mmio_addr[6:2] < (N >> 2)) begin
                    for (cn = 0; cn < 4; cn = cn + 1)
                        reg_rdata[cn*8 +: 8] = {3'd0, ctx[{mmio_addr[6:2], 2'b00} + cn]};
                end
            end
        endcase
    end

    // =================================================================== FSM
    integer q;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            mode_gen <= 1'b0;
            seed_r <= 32'd1;
            nsteps_r <= 32'd0;
            gen_len_r <= 32'd0;
            corpus_len_r <= 0;
            hp_s1 <= 5'd0;  hp_s2 <= 5'd4;  hp_g1 <= 5'd4;  hp_g2 <= 5'd0;
            hp_lr1 <= 5'd14; hp_lr2 <= 5'd8; hp_lre <= 5'd8; hp_lrb <= 5'd12;
            hp_clip <= 16'd32767;
            hp_flags <= 8'd0;
            steps_done <= 32'd0;
            correct_cnt <= 32'd0;
            sum_pt <= 64'd0;
            cycles <= 64'd0;
            rng <= 32'd1;
            err_r <= 1'b0;
            done_r <= 1'b0;
            for (q = 0; q < N; q = q + 1) ctx[q] <= 5'd0;
            for (q = 0; q < E; q = q + 1) esumv[q] <= 24'sd0;
            w1_we <= {L{1'b0}}; w2_we <= {L{1'b0}}; emb_we <= {E{1'b0}};
            corp_we <= 1'b0; gen_we <= 4'd0;
            w1_raddr <= 0; w2_raddr <= 0; emb_raddr <= 0; corp_raddr <= 0; gen_raddr <= 0;
            w1_waddr <= 0; w2_waddr <= 0; emb_waddr <= 0; corp_waddr <= 0; gen_waddr <= 0;
            w1_wdata <= 0; w2_wdata <= 0; emb_wdata <= 0; corp_wdata <= 0; gen_wdata <= 0;
            p1_v <= 1'b0; p2_v <= 1'b0;
            p1_row <= 0; p2_row <= 0; p1_blk <= 0; p2_blk <= 0; p1_dh <= 0; p2_dh <= 0;
            mmio_rvalid <= 1'b0;
            mmio_rdata <= 32'd0;
            mr_cnt <= 3'd0;
            mr_kind <= 3'd0;
            mr_lane <= 8'd0;
            issuing <= 1'b0;
            drain <= 3'd0;
            sub <= 8'd0;
            i_row <= 0; i_blk <= 0;
        end else begin
            // vychozi: zadne zapisy do pameti
            w1_we <= {L{1'b0}};
            w2_we <= {L{1'b0}};
            emb_we <= {E{1'b0}};
            corp_we <= 1'b0;
            gen_we <= 4'd0;
            mmio_rvalid <= 1'b0;

            // pipeline cteni posouva vzdy
            p2_v <= p1_v;
            p2_row <= p1_row;
            p2_blk <= p1_blk;
            p2_dh <= p1_dh;
            p1_v <= 1'b0;

            if (state != S_IDLE) cycles <= cycles + 64'd1;

            // ------------------------------------------------ MMIO zapis
            if (mmio_we && state == S_IDLE) begin
                if (mmio_addr[23:16] == 8'h00) begin
                    case (mmio_addr[7:0])
                        8'h08: begin
                            // CTRL: 1 INIT, 2 TRAIN, 3 GEN, 4 CLEAR_STATS
                            done_r <= 1'b0;
                            err_r <= 1'b0;
                            cycles <= 64'd0;
                            case (mmio_wdata[3:0])
                                4'd1: begin
                                    rng <= (seed_r == 32'd0) ? 32'd1 : seed_r;
                                    init_sel <= 2'd0;
                                    init_addr <= 16'd0;
                                    init_lane <= 8'd0;
                                    state <= S_INIT;
                                end
                                4'd2: begin
                                    if (corpus_len_r <= N || corpus_len_r > (1 << LOG2_CORPUS)) begin
                                        err_r <= 1'b1;
                                        done_r <= 1'b1;
                                    end else if (nsteps_r == 32'd0) begin
                                        done_r <= 1'b1;
                                    end else begin
                                        steps_left <= nsteps_r;
                                        mode_gen <= 1'b0;
                                        state <= S_T_POS;
                                    end
                                end
                                4'd3: begin
                                    if (gen_len_r == 32'd0) begin
                                        done_r <= 1'b1;
                                    end else begin
                                        gen_left <= gen_len_r;
                                        gen_ptr <= 0;
                                        mode_gen <= 1'b1;
                                        sub <= 8'd0;
                                        issuing <= 1'b1;
                                        state <= S_F_EMB;
                                    end
                                end
                                4'd4: begin
                                    steps_done <= 32'd0;
                                    correct_cnt <= 32'd0;
                                    sum_pt <= 64'd0;
                                    done_r <= 1'b1;
                                end
                                default: ;
                            endcase
                        end
                        8'h10: seed_r <= mmio_wdata;
                        8'h14: nsteps_r <= mmio_wdata;
                        8'h18: corpus_len_r <= mmio_wdata[LOG2_CORPUS:0];
                        8'h1C: gen_len_r <= mmio_wdata;
                        8'h20: begin
                            hp_s1 <= mmio_wdata[4:0];   hp_s2 <= mmio_wdata[12:8];
                            hp_g1 <= mmio_wdata[20:16]; hp_g2 <= mmio_wdata[28:24];
                        end
                        8'h24: begin
                            hp_lr1 <= mmio_wdata[4:0];   hp_lr2 <= mmio_wdata[12:8];
                            hp_lre <= mmio_wdata[20:16]; hp_lrb <= mmio_wdata[28:24];
                        end
                        8'h28: begin
                            hp_clip <= (mmio_wdata[15] ? 16'd32767 : mmio_wdata[15:0]);
                            hp_flags <= mmio_wdata[23:16];
                        end
                        8'h3C: rng <= (mmio_wdata == 32'd0) ? 32'd1 : mmio_wdata;
                        default: begin
                            if (mmio_addr[7] && mmio_addr[6:2] < (N >> 2))
                                for (q = 0; q < 4; q = q + 1)
                                    ctx[{mmio_addr[6:2], 2'b00} + q] <= mmio_wdata[q*8 +: 5];
                        end
                    endcase
                end else if (mmio_addr[23:22] == 2'b01) begin
                    // korpus 0x400000-0x7FFFFF
                    corp_waddr <= mmio_addr[LOG2_CORPUS-1:2];
                    corp_wdata <= mmio_wdata & 32'h1F1F_1F1F;
                    corp_we <= 1'b1;
                end else if (mmio_addr[23:22] == 2'b10) begin
                    // W1 0x800000-0xBFFFFF: index = addr[21:2]
                    w1_waddr <= mmio_addr[2 + LOG2_L +: W1_AW];
                    for (q = 0; q < L; q = q + 1) w1_wdata[q*16 +: 16] <= mmio_wdata[15:0];
                    w1_we <= {{(L-1){1'b0}}, 1'b1} << mmio_addr[2 +: LOG2_L];
                end else if (mmio_addr[23:20] == 4'hC) begin
                    w2_waddr <= mmio_addr[2 + LOG2_L +: W2_AW];
                    for (q = 0; q < L; q = q + 1) w2_wdata[q*16 +: 16] <= mmio_wdata[15:0];
                    w2_we <= {{(L-1){1'b0}}, 1'b1} << mmio_addr[2 +: LOG2_L];
                end else if (mmio_addr[23:16] == 8'hD0) begin
                    emb_waddr <= mmio_addr[2 + LOG2_E +: LOG2_V];
                    for (q = 0; q < E; q = q + 1) emb_wdata[q*16 +: 16] <= mmio_wdata[15:0];
                    emb_we <= {{(E-1){1'b0}}, 1'b1} << mmio_addr[2 +: LOG2_E];
                end
            end

            // ------------------------------------------------ MMIO cteni
            if (mr_cnt != 3'd0) begin
                mr_cnt <= mr_cnt - 3'd1;
                if (mr_cnt == 3'd1) begin
                    mmio_rvalid <= 1'b1;
                    case (mr_kind)
                        3'd1: mmio_rdata <= sx16(w1_rdata[mr_lane*16 +: 16]);
                        3'd2: mmio_rdata <= sx16(w2_rdata[mr_lane*16 +: 16]);
                        3'd3: mmio_rdata <= sx16(emb_rdata[mr_lane*16 +: 16]);
                        3'd4: mmio_rdata <= corp_rdata;
                        default: mmio_rdata <= gen_rdata;
                    endcase
                end
            end else if (mmio_re) begin
                if (mmio_addr[23:16] == 8'h00) begin
                    mmio_rdata <= reg_rdata;
                    mmio_rvalid <= 1'b1;
                end else if (state != S_IDLE) begin
                    mmio_rdata <= 32'hDEAD_BEEF;
                    mmio_rvalid <= 1'b1;
                end else if (mmio_addr[23:16] == 8'h01) begin
                    gen_raddr <= mmio_addr[LOG2_GEN-1:2];
                    mr_kind <= 3'd5; mr_cnt <= 3'd2;
                end else if (mmio_addr[23:22] == 2'b01) begin
                    corp_raddr <= mmio_addr[LOG2_CORPUS-1:2];
                    mr_kind <= 3'd4; mr_cnt <= 3'd2;
                end else if (mmio_addr[23:22] == 2'b10) begin
                    w1_raddr <= mmio_addr[2 + LOG2_L +: W1_AW];
                    mr_lane <= mmio_addr[2 +: LOG2_L];
                    mr_kind <= 3'd1; mr_cnt <= 3'd2;
                end else if (mmio_addr[23:20] == 4'hC) begin
                    w2_raddr <= mmio_addr[2 + LOG2_L +: W2_AW];
                    mr_lane <= mmio_addr[2 +: LOG2_L];
                    mr_kind <= 3'd2; mr_cnt <= 3'd2;
                end else if (mmio_addr[23:16] == 8'hD0) begin
                    emb_raddr <= mmio_addr[2 + LOG2_E +: LOG2_V];
                    mr_lane <= mmio_addr[2 +: LOG2_E];
                    mr_kind <= 3'd3; mr_cnt <= 3'd2;
                end else begin
                    mmio_rdata <= 32'd0;
                    mmio_rvalid <= 1'b1;
                end
            end

            // ================================================== stavovy automat
            case (state)
            S_IDLE: ;

            // ---------------------------------------------- INIT: vahy z RNG
            S_INIT: begin
                rng <= rng_nx;
                ibuf[init_lane*16 +: 16] <= (init_sel == 2'd2) ? init_e : init_w;
                if ((init_sel == 2'd2 && init_lane == E - 1) ||
                    (init_sel != 2'd2 && init_lane == L - 1)) begin
                    init_lane <= 8'd0;
                    state <= S_INIT_WR;
                end else begin
                    init_lane <= init_lane + 8'd1;
                end
            end
            S_INIT_WR: begin
                // ibuf je kompletni (posledni lane zapsana v minulem taktu)
                case (init_sel)
                    2'd0: begin w1_waddr <= init_addr[W1_AW-1:0]; w1_wdata <= ibuf; w1_we <= {L{1'b1}}; end
                    2'd1: begin w2_waddr <= init_addr[W2_AW-1:0]; w2_wdata <= ibuf; w2_we <= {L{1'b1}}; end
                    default: begin emb_waddr <= init_addr[LOG2_V-1:0]; emb_wdata <= ibuf[16*E-1:0]; emb_we <= {E{1'b1}}; end
                endcase
                state <= S_INIT;
                if (init_sel == 2'd0 && init_addr == W1_WORDS - 1) begin
                    init_sel <= 2'd1; init_addr <= 16'd0;
                end else if (init_sel == 2'd1 && init_addr == W2_WORDS - 1) begin
                    init_sel <= 2'd2; init_addr <= 16'd0;
                end else if (init_sel == 2'd2 && init_addr == V - 1) begin
                    state <= S_DONE;
                end else begin
                    init_addr <= init_addr + 16'd1;
                end
            end

            // ------------------------------------- TRAIN: nahodna pozice v korpusu
            S_T_POS: begin
                rng <= rng_nx;
                pos <= pos_next;
                sub <= 8'd0;
                issuing <= 1'b1;
                state <= S_T_FETCH;
            end
            S_T_FETCH: begin
                // cte tokeny pos-N .. pos (N+1 tokenu), p_blk = index
                if (issuing) begin
                    corp_raddr <= fetch_idx[LOG2_CORPUS-1:2];
                    p1_v <= 1'b1;
                    p1_blk <= {14'd0, fetch_idx[1:0]};
                    p1_row <= {8'd0, sub};
                    if (sub == N) issuing <= 1'b0;
                    else sub <= sub + 8'd1;
                end
                if (p2_v) begin
                    if (p2_row == N) target <= corp_rdata[p2_blk[1:0]*8 +: 5];
                    else ctx[p2_row[LOG2_N-1:0]] <= corp_rdata[p2_blk[1:0]*8 +: 5];
                    if (p2_row == N) begin
                        sub <= 8'd0;
                        issuing <= 1'b1;
                        state <= S_F_EMB;
                    end
                end
            end

            // ---------------------------------------------- FORWARD: embedding
            S_F_EMB: begin
                if (issuing) begin
                    emb_raddr <= ctx[sub[LOG2_N-1:0]];
                    p1_v <= 1'b1;
                    p1_row <= {8'd0, sub};
                    if (sub == N - 1) issuing <= 1'b0;
                    else sub <= sub + 8'd1;
                end
                if (p2_v) begin
                    // soucet stinovych embeddingu v okne; na konci okna x = clamp(sum >>> (8 + lg))
                    for (q = 0; q < E; q = q + 1) begin : femb_l
                        reg signed [23:0] sm;
                        sm = esumv[q] + {{8{emb_rdata[q*16 + 15]}}, emb_rdata[q*16 +: 16]};
                        if (femb_last) begin
                            xv[(femb_slot << LOG2_E) + q] <= win2x(sm, femb_lg);
                            esumv[q] <= 24'sd0;
                        end else begin
                            esumv[q] <= sm;
                        end
                    end
                    if (p2_row == N - 1) begin
                        i_row <= 0; i_blk <= 0;
                        issuing <= 1'b1;
                        acc <= 32'sd0;
                        hmax <= 24'd0;
                        state <= S_F_W1;
                    end
                end
            end

            // ------------------------------------------- FORWARD: h = W1b * x
            S_F_W1: begin
                if (issuing) begin
                    w1_raddr <= (i_row[LOG2_H-1:0] << D_BLK_LOG2) | i_blk[W1_AW-1:0];
                    p1_v <= 1'b1;
                    p1_row <= i_row;
                    p1_blk <= i_blk;
                    if (i_blk == (1 << D_BLK_LOG2) - 1) begin
                        i_blk <= 0;
                        if (i_row == H - 1) issuing <= 1'b0;
                        else i_row <= i_row + 16'd1;
                    end else begin
                        i_blk <= i_blk + 16'd1;
                    end
                end
                if (p2_v) begin
                    if (p2_blk == (1 << D_BLK_LOG2) - 1) begin
                        hv[p2_row[LOG2_H-1:0]] <= acc[23:0] + fw1_part[23:0];
                        if ($signed(acc + fw1_part) > $signed({8'd0, hmax}))
                            hmax <= acc[23:0] + fw1_part[23:0];
                        acc <= 32'sd0;
                        if (p2_row == H - 1) begin
                            i_blk <= 0;
                            state <= S_F_ACT;
                        end
                    end else begin
                        acc <= acc + fw1_part;
                    end
                end
            end

            // ----------------------------- FORWARD: a = clamp(h >>> s, 0, 127)
            S_F_ACT: begin
                if (i_blk == 0) s1dyn <= dyn_shift(hmax, hp_s1);
                // i_blk 0: vypocet posunu, 1..H/L: kusy po L
                if (i_blk != 0) begin
                    for (q = 0; q < L; q = q + 1) begin : act_l
                        reg signed [23:0] t;
                        t = hv[((i_blk - 1) << LOG2_L) + q] >>> s1dyn;
                        av[((i_blk - 1) << LOG2_L) + q] <= (t < 0) ? 7'd0 : ((t > 24'sd127) ? 7'd127 : t[6:0]);
                    end
                end
                if (i_blk == (1 << H_BLK_LOG2)) begin
                    i_row <= 0; i_blk <= 0;
                    issuing <= 1'b1;
                    acc <= 32'sd0;
                    state <= S_F_W2;
                end else begin
                    i_blk <= i_blk + 16'd1;
                end
            end

            // ------------------------------------------- FORWARD: z = W2b * a
            S_F_W2: begin
                if (issuing) begin
                    w2_raddr <= (i_row[LOG2_V-1:0] << H_BLK_LOG2) | i_blk[W2_AW-1:0];
                    p1_v <= 1'b1;
                    p1_row <= i_row;
                    p1_blk <= i_blk;
                    if (i_blk == (1 << H_BLK_LOG2) - 1) begin
                        i_blk <= 0;
                        if (i_row == V - 1) issuing <= 1'b0;
                        else i_row <= i_row + 16'd1;
                    end else begin
                        i_blk <= i_blk + 16'd1;
                    end
                end
                if (p2_v) begin
                    if (p2_blk == (1 << H_BLK_LOG2) - 1) begin
                        zv[p2_row[LOG2_V-1:0]] <= (acc + fw2_part) >>> hp_s2;
                        if (p2_row == 0 || (((acc + fw2_part) >>> hp_s2) > $signed({{8{zmax[23]}}, zmax}))) begin
                            zmax <= ((acc + fw2_part) >>> hp_s2);
                            argmax <= p2_row[4:0];
                        end
                        acc <= 32'sd0;
                        if (p2_row == V - 1) begin
                            sub <= 8'd0;
                            esum <= 23'd0;
                            state <= S_F_EXP;
                        end
                    end else begin
                        acc <= acc + fw2_part;
                    end
                end
            end

            // --------------------------------------- SOFTMAX: e = exp(zmax - z)
            S_F_EXP: begin : f_exp
                reg signed [24:0] d;
                reg [16:0] ex;
                d = $signed({zmax[23], zmax}) - $signed({zv[sub[4:0]][23], zv[sub[4:0]]});
                ex = exp_lut((d > 25'sd255) ? 8'd255 : d[7:0]);
                ev[sub[4:0]] <= ex;
                esum <= esum + {6'd0, ex};
                if (sub == V - 1) begin
                    sub <= 8'd0;
                    div_rem <= 33'd0;
                    div_q <= 33'd0;
                    state <= S_F_DIV;
                end else begin
                    sub <= sub + 8'd1;
                end
            end

            // ------------------------- SOFTMAX: rcp = floor(2^32 / esum), 33 taktu
            S_F_DIV: begin : f_div
                reg [33:0] rr;
                // bit dividendu 2^32: 1 jen pro nejvyssi bit (sub == 0)
                rr = {div_rem, (sub == 8'd0) ? 1'b1 : 1'b0};
                if (rr >= {11'd0, esum}) begin
                    div_rem <= rr[32:0] - {10'd0, esum};
                    div_q <= {div_q[31:0], 1'b1};
                end else begin
                    div_rem <= rr[32:0];
                    div_q <= {div_q[31:0], 1'b0};
                end
                if (sub == 8'd32) begin
                    sub <= 8'd0;
                    state <= S_F_PROB;
                end else begin
                    sub <= sub + 8'd1;
                end
            end

            // ---------------------- SOFTMAX: p = (e * rcp) >> 16, gradient gz
            S_F_PROB: begin : f_prob
                reg [33:0] pp;
                reg signed [39:0] g;
                if (sub == 8'd0) rcp <= div_q[16:0];
                // div_q je platny az po S_F_DIV; rcp pouzit od sub >= 1
                if (sub != 8'd0) begin
                    pp = {17'd0, ev[sub[4:0] - 5'd1]} * {17'd0, rcp};
                    pv[sub[4:0] - 5'd1] <= pp[32:16];
                    g = $signed({23'd0, pp[32:16]}) - (((sub[4:0] - 5'd1) == target && !mode_gen) ? 40'sd65536 : 40'sd0);
                    gzv[sub[4:0] - 5'd1] <= rsr40(g, hp_g1);
                    if (!mode_gen && (sub[4:0] - 5'd1) == target)
                        sum_pt <= sum_pt + {47'd0, pp[32:16]};
                end
                if (sub == V) begin
                    sub <= 8'd0;
                    if (mode_gen) begin
                        cum <= 17'd0;
                        found <= 1'b0;
                        state <= S_G_SAMP;
                    end else begin
                        steps_done <= steps_done + 32'd1;
                        if (argmax == target) correct_cnt <= correct_cnt + 32'd1;
                        for (q = 0; q < H; q = q + 1) dav[q] <= 32'sd0;
                        i_row <= 0; i_blk <= 0;
                        issuing <= 1'b1;
                        state <= S_B_W2;
                    end
                end else begin
                    sub <= sub + 8'd1;
                end
            end

            // ------------------- BACKWARD W2: da += gz*sign(w), w -= rsr(gz*a)
            S_B_W2: begin
                if (issuing) begin
                    w2_raddr <= (i_row[LOG2_V-1:0] << H_BLK_LOG2) | i_blk[W2_AW-1:0];
                    p1_v <= 1'b1;
                    p1_row <= i_row;
                    p1_blk <= i_blk;
                    if (i_blk == (1 << H_BLK_LOG2) - 1) begin
                        i_blk <= 0;
                        if (i_row == V - 1) issuing <= 1'b0;
                        else i_row <= i_row + 16'd1;
                    end else begin
                        i_blk <= i_blk + 16'd1;
                    end
                end
                if (p2_v) begin
                    w2_waddr <= (p2_row[LOG2_V-1:0] << H_BLK_LOG2) | p2_blk[W2_AW-1:0];
                    w2_wdata <= bw2_new;
                    w2_we <= {L{1'b1}};
                    for (q = 0; q < L; q = q + 1) begin
                        if (!w2_rdata[q*16 + 15])
                            dav[(p2_blk << LOG2_L) + q] <= dav[(p2_blk << LOG2_L) + q] + gzv[p2_row[LOG2_V-1:0]];
                        else
                            dav[(p2_blk << LOG2_L) + q] <= dav[(p2_blk << LOG2_L) + q] - gzv[p2_row[LOG2_V-1:0]];
                    end
                    if (p2_row == V - 1 && p2_blk == (1 << H_BLK_LOG2) - 1) begin
                        i_blk <= 0;
                        state <= S_B_DH;
                    end
                end
            end

            // ------------------------------ BACKWARD: dh = mask(a) * rsr(da, g2)
            S_B_DH: begin
                for (q = 0; q < L; q = q + 1) begin : dh_l
                    reg [15:0] jj;
                    reg signed [39:0] r;
                    jj = (i_blk << LOG2_L) + q;
                    r = rsr40($signed({{8{dav[jj][31]}}, dav[jj]}), hp_g2);
                    dhv[jj] <= (av[jj] != 7'd0 && av[jj] != 7'd127) ? r[31:0] : 32'sd0;
                end
                if (i_blk == (1 << H_BLK_LOG2) - 1) begin
                    for (q = 0; q < D; q = q + 1) dxv[q] <= 32'sd0;
                    i_row <= 0; i_blk <= 0;
                    issuing <= 1'b1;
                    drain <= 3'd0;
                    state <= S_B_W1;
                end else begin
                    i_blk <= i_blk + 16'd1;
                end
            end

            // ------------ BACKWARD W1: dx += dh*sign(w), w -= rsr(dh*x); radky dh=0 preskocit
            S_B_W1: begin
                if (issuing) begin
                    if (i_blk == 0 && dh_sel == 32'sd0) begin
                        // radek bez gradientu: zadna zmena vah ani dx
                        if (i_row == H - 1) issuing <= 1'b0;
                        else i_row <= i_row + 16'd1;
                    end else begin
                        w1_raddr <= (i_row[LOG2_H-1:0] << D_BLK_LOG2) | i_blk[W1_AW-1:0];
                        p1_v <= 1'b1;
                        p1_row <= i_row;
                        p1_blk <= i_blk;
                        p1_dh <= dh_sel;
                        if (i_blk == (1 << D_BLK_LOG2) - 1) begin
                            i_blk <= 0;
                            if (i_row == H - 1) issuing <= 1'b0;
                            else i_row <= i_row + 16'd1;
                        end else begin
                            i_blk <= i_blk + 16'd1;
                        end
                    end
                end
                if (p2_v) begin
                    w1_waddr <= (p2_row[LOG2_H-1:0] << D_BLK_LOG2) | p2_blk[W1_AW-1:0];
                    w1_wdata <= bw1_new;
                    w1_we <= {L{1'b1}};
                    for (q = 0; q < L; q = q + 1) begin
                        if (!w1_rdata[q*16 + 15])
                            dxv[(p2_blk << LOG2_L) + q] <= dxv[(p2_blk << LOG2_L) + q] + p2_dh;
                        else
                            dxv[(p2_blk << LOG2_L) + q] <= dxv[(p2_blk << LOG2_L) + q] - p2_dh;
                    end
                end
                // konec: nic se nevydava a pipeline je prazdna
                if (!issuing && !p1_v && !p2_v) begin
                    if (drain == 3'd1) begin
                        sub <= 8'd0;
                        drain <= 3'd0;
                        state <= S_B_EMB;
                    end else begin
                        drain <= drain + 3'd1;
                    end
                end
            end

            // ----------------- BACKWARD: embedding, pozice n postupne (RAW bezpecne)
            // drain: 0 adresa, 1-2 cekani na data, 3 zapis, 4 mezera (zapis probehne)
            S_B_EMB: begin
                case (drain)
                    3'd0: begin
                        emb_raddr <= ctx[sub[LOG2_N-1:0]];
                        drain <= 3'd1;
                    end
                    3'd1: drain <= 3'd2;
                    3'd2: drain <= 3'd3;
                    3'd3: begin
                        emb_waddr <= ctx[sub[LOG2_N-1:0]];
                        emb_wdata <= emb_new;
                        emb_we <= {E{1'b1}};
                        drain <= 3'd4;
                    end
                    default: begin
                        drain <= 3'd0;
                        if (sub == N - 1) begin
                            sub <= 8'd0;
                            if (steps_left == 32'd1) begin
                                state <= S_DONE;
                            end else begin
                                steps_left <= steps_left - 32'd1;
                                state <= S_T_POS;
                            end
                        end else begin
                            sub <= sub + 8'd1;
                        end
                    end
                endcase
            end

            // ------------------------------------- GEN: vzorkovani z p (Q16)
            S_G_SAMP: begin : g_samp
                reg [17:0] c2;
                if (sub == 8'd0) begin
                    rng <= rng_nx;
                    u16 <= rng_nx[15:0];
                    sub <= 8'd1;
                end else begin
                    c2 = {1'b0, cum} + {1'b0, pv[sub[4:0] - 5'd1]};
                    cum <= c2[16:0];
                    if (!found && c2 > {2'd0, u16}) begin
                        found <= 1'b1;
                        tok <= sub[4:0] - 5'd1;
                    end
                    if (sub == V) begin
                        if (!found && !(c2 > {2'd0, u16})) tok <= V - 1;
                        state <= S_G_EMIT;
                    end
                    sub <= sub + 8'd1;
                end
            end
            S_G_EMIT: begin
                gen_waddr <= gen_ptr[LOG2_GEN-1:2];
                gen_wdata <= {4{3'd0, tok}};
                gen_we <= 4'b0001 << gen_ptr[1:0];
                gen_ptr <= gen_ptr + 1'b1;
                for (q = 0; q < N - 1; q = q + 1) ctx[q] <= ctx[q + 1];
                ctx[N - 1] <= tok;
                sub <= 8'd0;
                if (gen_left == 32'd1) begin
                    state <= S_DONE;
                end else begin
                    gen_left <= gen_left - 32'd1;
                    issuing <= 1'b1;
                    state <= S_F_EMB;
                end
            end

            S_DONE: begin
                done_r <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
