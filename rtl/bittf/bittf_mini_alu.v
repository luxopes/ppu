// BitTF mini ALU: stejne funkce jako bittf_alu (bitove shodne), ale bez kombinacnich nasobicek:
// soucin pro funkce MUL* (mul_need) spocita predem sdilena postupna nasobicka v jadru z mul_a, mul_b
// a drzi ho na mul_p; deleni 66 taktu.
module bittf_mini_alu (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [7:0]  fn,
    input  wire [2:0]  tc,
    input  wire        clamp_en,
    input  wire        clamp_sym,
    input  wire [6:0]  cbits,
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [63:0] c,
    input  wire [63:0] r1,
    input  wire [63:0] r2,
    input  wire [31:0] imm,
    input  wire [31:0] i0,
    input  wire [31:0] i1,
    input  wire [31:0] rnd,
    output wire        mul_need,
    output reg  [63:0] mul_a,
    output reg  [63:0] mul_b,
    input  wire [63:0] mul_p,
    output reg         done,
    output reg  [63:0] y
);
`include "bittf_isa.vh"
`include "bittf_tables.vh"
`include "bittf_func.vh"

    wire signed [63:0] sa = a, sb = b, sc = c, sr1 = r1, sr2 = r2;
    wire signed [63:0] simm = {{32{imm[31]}}, imm};
    reg  signed [63:0] tcv;
    always @(*) begin
        case (tc)
            TC_ADD:     tcv = sc;
            TC_SUB:     tcv = -sc;
            TC_ADDHALF: tcv = sc >>> 1;
            TC_SUBHALF: tcv = -(sc >>> 1);
            default:    tcv = 64'sd0;
        endcase
    end
    wire signed [63:0] sh = simm + sr1 + tcv;

    // ---------------------------------------------------- sdileny posouvac (jediny v ALU)
    // su_y = s >= 0: (v + add) >>> min(s, smax), jinak v <<< min(-s, 63)
    //   RSR: f_rsr (smax 62, add = pul), SRD: f_srd (smax 32, add = rnd & maska),
    //   SHR: posun vpravo (smax 63, add 0; zaporne s jen u NRM -> vlevo), SHRF: zaporne s = 0
    localparam SU_RSR = 2'd0, SU_SRD = 2'd1, SU_SHR = 2'd2, SU_SHRF = 2'd3;
    reg  signed [63:0] su_v, su_s;
    reg  [1:0]         su_m;
    reg                r2a;            // RSR2ADD: druha polovina
    reg  signed [63:0] r2a_t;
    wire signed [63:0] pq_v;           // POW2Q: mocnina dvou pred posunem
    always @(*) begin
        su_m = SU_RSR;
        case (fn)
            E_RSR:        begin su_v = sa;  su_s = sh; end
            E_ADDRSR:     begin su_v = sb;  su_s = sh; end
            E_MULRSR,
            E_MULRELURSR: begin su_v = smp; su_s = sh; end
            E_SRD:        begin su_v = sa;  su_s = sh; su_m = SU_SRD; end
            E_SHRF:       begin su_v = sa;  su_s = sh; su_m = SU_SHRF; end
            E_NRMR:       begin su_v = sa;  su_s = sb; su_m = SU_SHR; end
            E_POW2Q:      begin su_v = pq_v; su_s = {58'd0, sh[5:0]}; su_m = SU_SHRF; end
            E_RSR2ADD:    begin su_v = r2a ? sb : sa; su_s = r2a ? sr2 : sr1; end
            default:      begin su_v = sa;  su_s = sh; end
        endcase
    end
    wire        su_right = !su_s[63] || su_m == SU_SHRF;
    wire [5:0]  su_rmax = su_m == SU_RSR ? 6'd62 : su_m == SU_SRD ? 6'd32 : 6'd63;
    wire [5:0]  su_sp = su_s[63] ? 6'd0 : (su_s > $signed({58'd0, su_rmax})) ? su_rmax : su_s[5:0];
    // -s > 63 (jako f_shl(v, -s)); pro s = min. int64 je -s zaporne -> posun (-s)[5:0] = 0
    wire        su_big = su_s < -64'sd63 && su_s != 64'sh8000000000000000;
    wire [5:0]  su_lp = su_big ? 6'd63 : (6'd0 - su_s[5:0]);
    wire [63:0] su_bit = 64'd1 << su_sp;
    wire signed [63:0] su_add = su_m == SU_RSR ? $signed(su_bit >> 1) :
                                su_m == SU_SRD ? ($signed({32'd0, rnd}) & $signed(su_bit - 64'd1)) : 64'sd0;
    wire signed [63:0] su_y = su_right ? ((su_v + su_add) >>> su_sp) : (su_v <<< su_lp);

    // NRMR: index do tabulky 1/sqrt (pred nasobenim, posouvac v rezimu SHR)
    reg signed [63:0] nrm_ix;
    always @(*) begin
        nrm_ix = su_y >>> 8;
        if (nrm_ix < 0) nrm_ix = 0;
        if (nrm_ix > 255) nrm_ix = 255;
    end

    // spolecny bitlen (BITLENI, NRMS, POW2Q z |a|)
    wire signed [63:0] pq_a = sa[63] ? -sa : sa;
    wire [6:0] bl = f_bitlen(fn == E_POW2Q ? pq_a : sa);
    // POW2Q (f_pow2q) bez posledniho posunu, ten dela posouvac
    wire signed [63:0] pq_b = (bl == 0) ? 64'sd1 : (64'sd1 <<< (bl - 7'd1));
    wire signed [63:0] pq_lo = 64'sd1 <<< sh[5:0];
    wire signed [63:0] pq_rr = $signed({32'd0, rnd});
    assign pq_v = (pq_a >= pq_lo) ? (((pq_rr & (pq_b - 64'sd1)) < pq_a - pq_b) ? (pq_b <<< 1) : pq_b)
                                  : (((pq_rr & (pq_lo - 64'sd1)) < pq_a) ? pq_lo : 64'sd0);
    wire signed [63:0] pq_res = sa[63] ? -su_y : (sa == 0 ? 64'sd0 : su_y);
    wire signed [63:0] nrms_s = $signed({57'd0, bl}) - 64'sd16;
    // nasobeni: operandy podle funkce, vysledek mp (dolnich 64 bitu)
    wire is_mul = fn == E_MUL || fn == E_MULRSR || fn == E_MULRELURSR || fn == E_MULSUB ||
                  fn == E_RELU2 || fn == E_MULADDI || fn == E_NRMR;
    assign mul_need = is_mul;
    wire signed [63:0] smp = mul_p;
    wire signed [63:0] rsq_prod = smp;

    // EXP8 / EXP16
    wire signed [63:0] dexp = sb - sa;
    wire [7:0] e8i = dexp[63] ? 8'd0 : (dexp > 255) ? 8'd255 : dexp[7:0];
    wire signed [63:0] dd16 = (dexp + 64'sd4) >>> 3;
    wire e16ok = !dd16[63] && dd16 < 512;

    wire signed [63:0] lion_t = (sb <<< 3) - sb + sc;
    wire signed [63:0] lion_u = lion_t > 0 ? 64'sd1 : lion_t < 0 ? -64'sd1 : 64'sd0;
    wire signed [63:0] relu_a = sa[63] ? 64'sd0 : sa;
    wire signed [63:0] relu_b = sb[63] ? 64'sd0 : sb;
    always @(*) begin
        case (fn)
            E_MULRELURSR: begin mul_a = sa; mul_b = relu_b; end
            E_RELU2:      begin mul_a = relu_a; mul_b = relu_a; end
            E_MULADDI:    begin mul_a = sa; mul_b = simm; end
            E_NRMR:       begin mul_a = simm; mul_b = {40'd0, rsq_lut(nrm_ix[7:0])}; end
            default:      begin mul_a = sa; mul_b = sb; end
        endcase
    end

    // ---------------------------------------------------- jedna hlavni scitacka: ad_a +/- ad_b
    reg  signed [63:0] ad_a, ad_b;
    reg                ad_sub;
    wire               oh_hit = {32'd0, i0} == b;
    always @(*) begin
        ad_a = sa; ad_b = sb; ad_sub = 1'b0;
        case (fn)
            E_SUB:     ad_sub = 1'b1;
            E_ADDRSR:  ad_b = su_y;
            E_MULSUB:  begin ad_a = smp; ad_b = sc; ad_sub = 1'b1; end
            E_BITLENI: begin ad_a = {57'd0, bl}; ad_b = simm; end
            E_OHSUB:   begin ad_b = oh_hit ? (64'sd1 <<< imm[5:0]) : 64'sd0; ad_sub = 1'b1; end
            E_RSR2ADD: begin ad_a = r2a_t; ad_b = su_y; end
            E_LIONQ:   begin ad_b = (lion_t == 0) ? 64'sd0 : (64'sd1 <<< r1[5:0]); ad_sub = lion_t > 0; end
            E_MULADDI: ad_a = smp;
            E_NRMR:    begin ad_a = smp; ad_b = 64'sd32768; end
            default: ;
        endcase
    end
    wire signed [63:0] ad_y = ad_a + (ad_sub ? ~ad_b : ad_b) + {63'd0, ad_sub};

    reg signed [63:0] res;
    always @(*) begin
        case (fn)
            E_COPY:       res = sa;
            E_ADD:        res = ad_y;
            E_SUB:        res = ad_y;
            E_MUL:        res = smp;
            E_RSR:        res = su_y;
            E_ADDRSR:     res = ad_y;
            E_MULRSR:     res = su_y;
            E_MULRELURSR: res = su_y;
            E_MULSUB:     res = ad_y;
            E_SRD:        res = su_y;
            E_POW2Q:      res = pq_res;
            E_BITLENI:    res = ad_y;
            E_SEXPR:      res = sh;
            E_NRMS:       res = nrms_s + (nrms_s & 64'sd1);
            E_NRMR:       res = ad_y >>> 16;
            E_EXP8:       res = {48'd0, exp8_lut(e8i)};
            E_EXP16:      res = e16ok ? {47'd0, exp16_lut(dd16[8:0])} : 64'sd0;
            E_OHSUB:      res = ad_y;
            E_ONEHOT:     res = ({32'd0, i0} == a) ? 64'sd1 : 64'sd0;
            E_RELU2:      res = smp;
            E_RSR2ADD:    res = ad_y;
            E_LIONQ:      res = ad_y;
            E_MULADDI:    res = ad_y;
            E_RNG:        res = {32'd0, rnd};
            E_SHRF:       res = su_y;
            E_LOBITS:     res = sa & ((64'sd1 <<< imm[5:0]) - 64'sd1);
            default:      res = 64'sd0;
        endcase
    end

    // orezani
    wire signed [63:0] chi = (64'sd1 <<< (cbits - 7'd1)) - 64'sd1;
    wire signed [63:0] clo = clamp_sym ? -chi : -chi - 64'sd1;
    function signed [63:0] clampf(input signed [63:0] v);
        begin
            if (!clamp_en) clampf = v;
            else if (v > chi) clampf = chi;
            else if (v < clo) clampf = clo;
            else clampf = v;
        end
    endfunction

    // deleni: restoring, bez znamenka, 64 bitu
    wire is_div = fn == E_RCP || fn == E_MEANDIV || fn == E_MODU;
    // jediny orezavac: hodnota pred orezanim podle toho, co se v tomto taktu dokoncuje
    reg  signed [63:0] ypre;
    wire signed [63:0] y_c = clampf(ypre);
    reg  [63:0] dv_q, dv_r, dv_d;
    reg  [6:0]  dv_n;
    reg         dv_busy, dv_neg;
    reg  [7:0]  dv_fn;
    wire [64:0] dv_rs = {dv_r, dv_q[63]};
    wire [64:0] dv_t = dv_rs - {1'b0, dv_d};
    wire signed [63:0] md_num = (sa <<< 1) + simm;
    always @(*) begin
        if (r2a || (start && !is_div && fn != E_RSR2ADD))
            ypre = res;
        else if (start && is_div)
            ypre = 64'sd0;                      // RCP se zapornym nebo nulovym delitelem
        else
            case (dv_fn)
                E_MODU:    ypre = dv_r;
                E_MEANDIV: ypre = dv_neg ? -$signed(dv_q) - ((dv_r != 0) ? 64'sd1 : 64'sd0) : dv_q;
                default:   ypre = dv_q;
            endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            dv_busy <= 1'b0;
            r2a <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && fn == E_RSR2ADD) begin
                r2a_t <= su_y;          // f_rsr(sa, sr1)
                r2a <= 1'b1;
            end else if (r2a) begin
                y <= y_c;               // + f_rsr(sb, sr2)
                done <= 1'b1;
                r2a <= 1'b0;
            end else if (start && !is_div) begin
                y <= y_c;
                done <= 1'b1;
            end else if (start) begin
                dv_fn <= fn;
                dv_n <= 7'd64;
                dv_neg <= 1'b0;
                dv_r <= 64'd0;
                case (fn)
                    E_RCP: begin
                        dv_q <= (64'd1 << imm[5:0]) + {sa[63], sa[63:1]};
                        dv_d <= a;
                    end
                    E_MEANDIV: begin
                        dv_q <= md_num[63] ? -md_num : md_num;
                        dv_neg <= md_num[63];
                        dv_d <= simm <<< 1;
                    end
                    default: begin
                        dv_q <= a;
                        dv_d <= {32'd0, imm};
                    end
                endcase
                if (fn == E_RCP && (sa[63] || sa == 0)) begin
                    y <= y_c;
                    done <= 1'b1;
                end else
                    dv_busy <= 1'b1;
            end else if (dv_busy) begin
                if (dv_n != 0) begin
                    // jeden krok: posun {r,q} vlevo, odecist delitel
                    if (!dv_t[64]) begin
                        dv_r <= dv_t[63:0];
                        dv_q <= {dv_q[62:0], 1'b1};
                    end else begin
                        dv_r <= dv_rs[63:0];
                        dv_q <= {dv_q[62:0], 1'b0};
                    end
                    dv_n <= dv_n - 7'd1;
                end else begin
                    dv_busy <= 1'b0;
                    done <= 1'b1;
                    y <= y_c;
                end
            end
        end
    end
endmodule
