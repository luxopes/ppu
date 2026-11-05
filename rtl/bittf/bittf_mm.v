// BitTF v2: maticove nasobeni D[i2][i1][i0] = sum_k A(i2,i1,k) * B(i2,i0,k) na poli TM x TN PE.
// Dlazdice TM radku x TN sloupcu, k po blocich KC. Bloky A a B se nacitaji proudy (port A a B,
// az LVL prvku na takt podel souvisle osy) do dvojitych bufferu, takze nacitani dalsiho bloku bezi
// soubezne s vypoctem. Hotova dlazdice se presune do vystupniho bufferu a zapisuje se (port D) behem
// vypoctu dalsi. PE: operandy 18 bitu se znamenkem (rezim SIGN: +-1 ze stinu), akumulator 48 bitu.
module bittf_mm #(
    parameter TM = 32,
    parameter TN = 32,
    parameter KC = 64,
    parameter LVL = 32,
    parameter VL = 8,
    parameter LB = 64
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    // instrukce
    input  wire [31:0]       n0, n1, n2, nk,
    input  wire [31:0]       a_base, b_base, d_base,
    input  wire [1:0]        a_lw, b_lw, d_lw,
    input  wire              a_uns, b_uns, d_uns,
    input  wire              b_sign,
    input  wire signed [63:0] as1, as2, as3, bs0, bs2, bs3, ds0, ds1, ds2,
    // port A, B (cteni)
    output wire              ra_valid, rb_valid,
    input  wire              ra_ready, rb_ready,
    output wire [25:0]       ra_line, rb_line,
    input  wire              sa_valid, sb_valid,
    input  wire [LB*8-1:0]   sa_data, sb_data,
    // zapis vysledku (k bittf_wstream)
    output reg               w_start,
    output reg  [31:0]       w_base,
    output reg  [127:0]      w_cnt,
    output reg  [255:0]      w_str,
    output reg               w_vec,
    output reg               w_valid,
    input  wire              w_ready,
    output reg  [VL*64-1:0]  w_data,
    output reg               done,
    output reg               err_fit
);
    localparam AW = 18;

    // ------------------------------------------------------------ nacitani bloku
    // Buffery A (TM radku x KC) a B (TN sloupcu x KC), dve sady. Kvuli syntéze na FPGA jsou to
    // samostatne registry v generate (bez zapisu s promennym indexem): takt proudu se jednim sdilenym
    // rotatorem posune na pozice (podel k nebo podel radku) a kazdy prvek ma vlastni povoleni zapisu.
    reg [1:0] full;                      // sada pripravena k vypoctu
    reg ld_set, cp_set;
    // iterace nacitani
    reg [31:0] l_i2, l_rb, l_cb, l_kc;   // indexy dlazdice (rb, cb v prvcich), kc v prvcich
    reg l_run, l_active;
    reg signed [63:0] l_aoff, l_boff;    // offset zacatku bloku v prvcich
    reg [31:0] l_rn, l_cn, l_kn;
    reg a_alongk, b_alongk, a_vec, b_vec;
    reg [31:0] a_p0, a_p1, b_p0, b_p1, a_c0, b_c0;
    reg a_done, b_done;
    reg st_start;

    wire [31:0] rows_left = n1 - l_rb, cols_left = n0 - l_cb, k_left = nk - l_kc;
    wire [31:0] rn_c = rows_left > TM ? TM : rows_left;
    wire [31:0] cn_c = cols_left > TN ? TN : cols_left;
    wire [31:0] kn_c = k_left > KC ? KC : k_left;

    // proudy
    reg [127:0] a_cnt, b_cnt;
    reg [255:0] a_str, b_str;
    wire a_ov, b_ov;
    wire [LVL*64-1:0] a_od, b_od;
    wire [6:0] a_on, b_on;
    wire a_busy, b_busy;
    wire a_take = a_ov && !a_done;
    wire b_take = b_ov && !b_done;
    bittf_stream #(.VL(LVL), .LB(LB)) sta (
        .clk(clk), .rst(rst), .start(st_start), .base(a_base + (l_aoff[31:0] << a_lw)), .lw(a_lw), .uns(a_uns),
        .vec(a_vec), .cnt(a_cnt), .str(a_str), .gath(1'b0), .gscale(64'sd0), .gidx_valid(1'b0), .gidx_ready(),
        .gidx(64'sd0), .rq_valid(ra_valid), .rq_ready(ra_ready), .rq_line(ra_line), .rs_valid(sa_valid),
        .rs_data(sa_data), .o_valid(a_ov), .o_ready(a_take), .o_data(a_od), .o_n(a_on), .busy(a_busy));
    bittf_stream #(.VL(LVL), .LB(LB)) stb (
        .clk(clk), .rst(rst), .start(st_start), .base(b_base + (l_boff[31:0] << b_lw)), .lw(b_lw), .uns(b_uns),
        .vec(b_vec), .cnt(b_cnt), .str(b_str), .gath(1'b0), .gscale(64'sd0), .gidx_valid(1'b0), .gidx_ready(),
        .gidx(64'sd0), .rq_valid(rb_valid), .rq_ready(rb_ready), .rq_line(rb_line), .rs_valid(sb_valid),
        .rs_data(sb_data), .o_valid(b_ov), .o_ready(b_take), .o_data(b_od), .o_n(b_on), .busy(b_busy));

    function [AW-1:0] bval(input [63:0] v);
        bval = b_sign ? (v[63] ? {AW{1'b1}} : {{(AW-1){1'b0}}, 1'b1}) : v[AW-1:0];
    endfunction

    // ------------------------------------------------------------ vypocet
    reg signed [47:0] acc [0:TM-1][0:TN-1];
    reg [31:0] c_k, c_kn;
    reg c_run, c_last;                   // posledni blok k dlazdice
    reg [31:0] c_rn, c_cn;
    reg signed [63:0] c_doff;
    // fronta informaci o blocich (nacitani -> vypocet): kn, posledni, rn, cn, doff
    reg [31:0] q_kn [0:1], q_rn [0:1], q_cn [0:1];
    reg q_last [0:1];
    reg signed [63:0] q_doff [0:1];

    // vystupni buffer (registry v generate, cteni: vyber radku a pak sloupcu)
    wire [TM*TN*48-1:0] ob_flat;
    reg o_full;
    reg [31:0] o_r, o_c, o_rn, o_cn;
    reg signed [63:0] o_doff;
    reg o_started;

    integer r, cc, l;
    reg all_loaded, all_done_c;

    // iterace dalsi dlazdice / bloku (nacitani)
    task l_next;
        begin
            if (l_kc + KC < nk) begin
                l_kc <= l_kc + KC;
                l_aoff <= l_aoff + as3 * KC;
                l_boff <= l_boff + bs3 * KC;
            end else begin
                l_kc <= 0;
                if (l_cb + TN < n0) begin
                    l_cb <= l_cb + TN;
                    l_aoff <= l_aoff - as3 * (l_kc);
                    l_boff <= l_boff - bs3 * (l_kc) + bs0 * TN;
                end else begin
                    l_cb <= 0;
                    if (l_rb + TM < n1) begin
                        l_rb <= l_rb + TM;
                        l_aoff <= l_aoff - as3 * (l_kc) + as1 * TM;
                        l_boff <= l_boff - bs3 * (l_kc) - bs0 * l_cb;
                    end else begin
                        l_rb <= 0;
                        if (l_i2 + 1 < n2) begin
                            l_i2 <= l_i2 + 1;
                            l_aoff <= l_aoff - as3 * (l_kc) - as1 * l_rb + as2;
                            l_boff <= l_boff - bs3 * (l_kc) - bs0 * l_cb + bs2;
                        end else
                            l_run <= 1'b0;
                    end
                end
            end
        end
    endtask

    wire signed [63:0] l_doff = ds2 * l_i2 + ds1 * l_rb + ds0 * l_cb;

    always @(posedge clk) begin
        if (rst) begin
            l_run <= 1'b0;
            l_active <= 1'b0;
            c_run <= 1'b0;
            full <= 2'b00;
            o_full <= 1'b0;
            done <= 1'b0;
            w_valid <= 1'b0;
            w_start <= 1'b0;
            st_start <= 1'b0;
            err_fit <= 1'b0;
        end else begin
            done <= 1'b0;
            st_start <= 1'b0;
            w_start <= 1'b0;
            if (start) begin
                l_i2 <= 0; l_rb <= 0; l_cb <= 0; l_kc <= 0;
                l_aoff <= 0; l_boff <= 0;
                l_run <= 1'b1;
                l_active <= 1'b0;
                ld_set <= 1'b0;
                cp_set <= 1'b0;
                full <= 2'b00;
                c_run <= 1'b0;
                o_full <= 1'b0;
                o_started <= 1'b0;
                err_fit <= 1'b0;
                for (r = 0; r < TM; r = r + 1)
                    for (cc = 0; cc < TN; cc = cc + 1)
                        acc[r][cc] <= 48'sd0;
            end else begin
                // --- nacitani: zahajeni bloku do volne sady
                if (l_run && !l_active && !full[ld_set]) begin
                    l_rn <= rn_c; l_cn <= cn_c; l_kn <= kn_c;
                    a_alongk <= (as3 == 1) || (as1 != 1);
                    b_alongk <= (bs3 == 1) || (bs0 != 1);
                    a_vec <= (as3 == 1) || (as1 == 1);
                    b_vec <= (bs3 == 1) || (bs0 == 1);
                    if ((as3 == 1) || (as1 != 1)) begin
                        a_cnt <= {32'd1, 32'd1, rn_c, kn_c}; a_str <= {64'sd0, 64'sd0, as1, as3}; a_c0 <= kn_c;
                    end else begin
                        a_cnt <= {32'd1, 32'd1, kn_c, rn_c}; a_str <= {64'sd0, 64'sd0, as3, as1}; a_c0 <= rn_c;
                    end
                    if ((bs3 == 1) || (bs0 != 1)) begin
                        b_cnt <= {32'd1, 32'd1, cn_c, kn_c}; b_str <= {64'sd0, 64'sd0, bs0, bs3}; b_c0 <= kn_c;
                    end else begin
                        b_cnt <= {32'd1, 32'd1, kn_c, cn_c}; b_str <= {64'sd0, 64'sd0, bs3, bs0}; b_c0 <= cn_c;
                    end
                    q_kn[ld_set] <= kn_c; q_rn[ld_set] <= rn_c; q_cn[ld_set] <= cn_c;
                    q_last[ld_set] <= l_kc + KC >= nk;
                    q_doff[ld_set] <= l_doff;
                    a_p0 <= 0; a_p1 <= 0; b_p0 <= 0; b_p1 <= 0;
                    a_done <= 1'b0; b_done <= 1'b0;
                    st_start <= 1'b1;
                    l_active <= 1'b1;
                end
                if (l_active && !st_start) begin
                    if (a_take) begin
                        if (a_p0 + a_on >= a_c0) begin
                            a_p0 <= 0;
                            a_p1 <= a_p1 + 1;
                            if (a_p1 + 1 >= (a_alongk ? l_rn : l_kn)) a_done <= 1'b1;
                        end else
                            a_p0 <= a_p0 + a_on;
                    end
                    if (b_take) begin
                        if (b_p0 + b_on >= b_c0) begin
                            b_p0 <= 0;
                            b_p1 <= b_p1 + 1;
                            if (b_p1 + 1 >= (b_alongk ? l_cn : l_kn)) b_done <= 1'b1;
                        end else
                            b_p0 <= b_p0 + b_on;
                    end
                    if (a_done && b_done) begin
                        full[ld_set] <= 1'b1;
                        ld_set <= ~ld_set;
                        l_active <= 1'b0;
                        l_next;
                    end
                end
                // --- vypocet
                if (!c_run && full[cp_set]) begin
                    c_run <= 1'b1;
                    c_k <= 0;
                    c_kn <= q_kn[cp_set];
                    c_last <= q_last[cp_set];
                    c_rn <= q_rn[cp_set];
                    c_cn <= q_cn[cp_set];
                    c_doff <= q_doff[cp_set];
                end
                // akumulace (3. stupen pipeline: vyber operandu -> soucin -> soucet)
                if (p_v2)
                    for (r = 0; r < TM; r = r + 1)
                        for (cc = 0; cc < TN; cc = cc + 1)
                            acc[r][cc] <= acc[r][cc] + prod[r][cc];
                if (c_run) begin
                    if (c_k < c_kn) begin
                        c_k <= c_k + 1;
                    end else if ((!c_last || !o_full) && !p_v1 && !p_v2) begin
                        // blok hotov
                        full[cp_set] <= 1'b0;
                        cp_set <= ~cp_set;
                        c_run <= 1'b0;
                        if (c_last) begin
                            for (r = 0; r < TM; r = r + 1)
                                for (cc = 0; cc < TN; cc = cc + 1)
                                    acc[r][cc] <= 48'sd0;
                            o_full <= 1'b1;
                            o_started <= 1'b0;
                            o_rn <= c_rn;
                            o_cn <= c_cn;
                            o_doff <= c_doff;
                            o_r <= 0;
                            o_c <= 0;
                        end
                    end
                end
                // --- zapis vystupni dlazdice
                if (w_valid && w_ready) w_valid <= 1'b0;
                if (o_full && !o_started && !w_valid) begin
                    w_start <= 1'b1;
                    w_base <= d_base + (o_doff[31:0] << d_lw);
                    w_cnt <= {32'd1, 32'd1, o_rn, o_cn};
                    w_str <= {64'sd0, 64'sd0, ds1, ds0};
                    w_vec <= ds0 == 1;
                    o_started <= 1'b1;
                end else if (o_full && o_started && !w_start && (!w_valid || w_ready)) begin
                    for (l = 0; l < VL; l = l + 1)
                        w_data[64 * l +: 64] <= {{16{o_lane[l][47]}}, o_lane[l]};
                    for (l = 0; l < VL; l = l + 1)
                        if (o_c + l < o_cn && !fits(o_lane[l])) err_fit <= 1'b1;
                    w_valid <= 1'b1;
                    if (o_c + (ds0 == 1 ? VL : 1) >= o_cn) begin
                        o_c <= 0;
                        if (o_r + 1 >= o_rn) o_full <= 1'b0;
                        else o_r <= o_r + 1;
                    end else
                        o_c <= o_c + (ds0 == 1 ? VL : 1);
                end
                // konec instrukce
                if (started_any && !l_run && !l_active && full == 2'b00 && !c_run && !o_full && !w_valid && !w_start)
                    done <= 1'b1;
            end
        end
    end


    // ------------------------------------------------------------ buffery A/B (generate)
    wire ld_ok = l_active && !st_start;
    wire [LVL*AW-1:0] a_lanes, b_lanes;
    genvar gl, gs, gr, gk;
    generate
        for (gl = 0; gl < LVL; gl = gl + 1) begin : LANES
            assign a_lanes[gl * AW +: AW] = (gl < a_on) ? a_od[64 * gl +: AW] : {AW{1'b0}};
            assign b_lanes[gl * AW +: AW] = (gl < b_on) ? bval(b_od[64 * gl +: 64]) : {AW{1'b0}};
        end
    endgenerate
    // posun taktu na pozice: podel k (KC pozic) nebo podel radku/sloupcu (TM/TN pozic)
    wire [(KC+LVL)*AW-1:0] a_wk2 = {{(KC*AW){1'b0}}, a_lanes} << (a_p0[5:0] * AW);
    wire [(KC+LVL)*AW-1:0] b_wk2 = {{(KC*AW){1'b0}}, b_lanes} << (b_p0[5:0] * AW);
    wire [(TM+LVL)*AW-1:0] a_wr2 = {{(TM*AW){1'b0}}, a_lanes} << (a_p0[5:0] * AW);
    wire [(TN+LVL)*AW-1:0] b_wr2 = {{(TN*AW){1'b0}}, b_lanes} << (b_p0[5:0] * AW);
    wire [KC*AW-1:0] a_wk = a_wk2[KC*AW-1:0], b_wk = b_wk2[KC*AW-1:0];
    wire [TM*AW-1:0] a_wr = a_wr2[TM*AW-1:0];
    wire [TN*AW-1:0] b_wr = b_wr2[TN*AW-1:0];
    wire [31:0] a_end = a_p0 + {25'd0, a_on}, b_end = b_p0 + {25'd0, b_on};
    // povoleni: podel k -> radek a_p1, k v [a_p0, a_end); jinak k = a_p1, radek v [a_p0, a_end)
    wire [TM-1:0] a_rsel, a_rmask;
    wire [TN-1:0] b_rsel, b_rmask;
    wire [KC-1:0] a_ksel, a_kmask, b_ksel, b_kmask;
    generate
        for (gr = 0; gr < TM; gr = gr + 1) begin : AROW
            assign a_rsel[gr] = a_p1 == gr;
            assign a_rmask[gr] = gr >= a_p0 && gr < a_end;
        end
        for (gr = 0; gr < TN; gr = gr + 1) begin : BROW
            assign b_rsel[gr] = b_p1 == gr;
            assign b_rmask[gr] = gr >= b_p0 && gr < b_end;
        end
        for (gk = 0; gk < KC; gk = gk + 1) begin : KSEL
            assign a_ksel[gk] = a_p1 == gk;
            assign a_kmask[gk] = gk >= a_p0 && gk < a_end;
            assign b_ksel[gk] = b_p1 == gk;
            assign b_kmask[gk] = gk >= b_p0 && gk < b_end;
        end
    endgenerate
    wire [1:0] a_we = {2{ld_ok && a_take}} & {ld_set, !ld_set};
    wire [1:0] b_we = {2{ld_ok && b_take}} & {ld_set, !ld_set};

    // operandy vypoctu: radek/sloupec pro aktualni k (vybrana sada cp_set)
    wire [AW-1:0] a_sel [0:TM-1];
    wire [AW-1:0] b_sel [0:TN-1];
    generate
        for (gs = 0; gs < 2; gs = gs + 1) begin : AS
            for (gr = 0; gr < TM; gr = gr + 1) begin : R
                wire [KC*AW-1:0] row;
                for (gk = 0; gk < KC; gk = gk + 1) begin : K
                    reg [AW-1:0] q;
                    always @(posedge clk)
                        if (a_we[gs] && (a_alongk ? (a_rsel[gr] && a_kmask[gk]) : (a_ksel[gk] && a_rmask[gr])))
                            q <= a_alongk ? a_wk[gk * AW +: AW] : a_wr[gr * AW +: AW];
                    assign row[gk * AW +: AW] = q;
                end
            end
        end
        for (gs = 0; gs < 2; gs = gs + 1) begin : BS
            for (gr = 0; gr < TN; gr = gr + 1) begin : R
                wire [KC*AW-1:0] row;
                for (gk = 0; gk < KC; gk = gk + 1) begin : K
                    reg [AW-1:0] q;
                    always @(posedge clk)
                        if (b_we[gs] && (b_alongk ? (b_rsel[gr] && b_kmask[gk]) : (b_ksel[gk] && b_rmask[gr])))
                            q <= b_alongk ? b_wk[gk * AW +: AW] : b_wr[gr * AW +: AW];
                    assign row[gk * AW +: AW] = q;
                end
            end
        end
        for (gr = 0; gr < TM; gr = gr + 1) begin : ASEL
            assign a_sel[gr] = cp_set ? AS[1].R[gr].row[c_k[5:0] * AW +: AW] : AS[0].R[gr].row[c_k[5:0] * AW +: AW];
        end
        for (gr = 0; gr < TN; gr = gr + 1) begin : BSEL
            assign b_sel[gr] = cp_set ? BS[1].R[gr].row[c_k[5:0] * AW +: AW] : BS[0].R[gr].row[c_k[5:0] * AW +: AW];
        end
    endgenerate

    // pipeline: 1. registr operandu, 2. soucin (DSP), 3. akumulace (v hlavnim bloku)
    wire p_issue = c_run && c_k < c_kn && !start;
    reg p_v1, p_v2;
    reg signed [AW-1:0] a_op [0:TM-1];
    reg signed [AW-1:0] b_op [0:TN-1];
    reg signed [2*AW-1:0] prod [0:TM-1][0:TN-1];
    integer pr, pc;
    always @(posedge clk) begin
        if (rst || start) begin
            p_v1 <= 1'b0;
            p_v2 <= 1'b0;
        end else begin
            p_v1 <= p_issue;
            p_v2 <= p_v1;
        end
        if (p_issue) begin
            for (pr = 0; pr < TM; pr = pr + 1) a_op[pr] <= a_sel[pr];
            for (pc = 0; pc < TN; pc = pc + 1) b_op[pc] <= b_sel[pc];
        end
        if (p_v1)
            for (pr = 0; pr < TM; pr = pr + 1)
                for (pc = 0; pc < TN; pc = pc + 1)
                    prod[pr][pc] <= a_op[pr] * b_op[pc];
    end

    // vystupni buffer: kopie akumulatoru pri dokonceni dlazdice
    wire o_copy = c_run && !(c_k < c_kn) && (!c_last || !o_full) && !p_v1 && !p_v2 && c_last && !start;
    generate
        for (gr = 0; gr < TM; gr = gr + 1) begin : OR
            for (gk = 0; gk < TN; gk = gk + 1) begin : OC
                reg [47:0] q;
                always @(posedge clk) if (o_copy) q <= acc[gr][gk];
                assign ob_flat[(gr * TN + gk) * 48 +: 48] = q;
            end
        end
    endgenerate
    wire [(TN+VL)*48-1:0] o_row = {{(VL*48){1'b0}}, ob_flat[o_r[4:0] * (TN * 48) +: TN * 48]};
    wire signed [47:0] o_lane [0:VL-1];
    generate
        for (gl = 0; gl < VL; gl = gl + 1) begin : OL
            assign o_lane[gl] = o_row[(o_c[5:0] + gl) * 48 +: 48];
        end
    endgenerate

    reg started_any;
    always @(posedge clk) begin
        if (rst || done) started_any <= 1'b0;
        else if (start) started_any <= 1'b1;
    end

    function fits(input signed [47:0] v);
        case (d_lw)
            2'd0: fits = d_uns ? v[47:8] == 0 : (v[47:7] == 0 || &v[47:7]);
            2'd1: fits = d_uns ? v[47:16] == 0 : (v[47:15] == 0 || &v[47:15]);
            2'd2: fits = d_uns ? v[47:32] == 0 : (v[47:31] == 0 || &v[47:31]);
            default: fits = 1'b1;
        endcase
    endfunction
endmodule
