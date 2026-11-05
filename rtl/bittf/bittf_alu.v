// BitTF ALU: funkce instrukce EW pro jeden prvek (docs/bittf_isa.md, reference sw/bittf/isa/bt_isa.c).
// Vsechny funkce trvaji 1 takt, deleni (RCP, MEANDIV, MODU) 66 taktu. Genericky Verilog bez primitiv.
module bittf_alu (
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

    // NRMR: index do tabulky 1/sqrt
    reg signed [63:0] nrm_mm, nrm_ix;
    always @(*) begin
        if (!sb[63])
            nrm_mm = sa >>> ((sb > 63) ? 6'd63 : sb[5:0]);
        else
            nrm_mm = f_shl(sa, -sb);
        nrm_ix = nrm_mm >>> 8;
        if (nrm_ix < 0) nrm_ix = 0;
        if (nrm_ix > 255) nrm_ix = 255;
    end
    wire signed [63:0] rsq_prod = simm * $signed({40'd0, rsq_lut(nrm_ix[7:0])});

    // EXP8 / EXP16
    wire signed [63:0] dexp = sb - sa;
    wire [7:0] e8i = dexp[63] ? 8'd0 : (dexp > 255) ? 8'd255 : dexp[7:0];
    wire signed [63:0] dd16 = (sb - sa + 64'sd4) >>> 3;
    wire e16ok = !dd16[63] && dd16 < 512;

    wire signed [63:0] lion_t = sb * 7 + sc;
    wire signed [63:0] lion_u = lion_t > 0 ? 64'sd1 : lion_t < 0 ? -64'sd1 : 64'sd0;
    wire signed [63:0] relu_a = sa[63] ? 64'sd0 : sa;
    wire signed [63:0] relu_b = sb[63] ? 64'sd0 : sb;

    reg signed [63:0] res;
    always @(*) begin
        case (fn)
            E_COPY:       res = sa;
            E_ADD:        res = sa + sb;
            E_SUB:        res = sa - sb;
            E_MUL:        res = sa * sb;
            E_RSR:        res = f_rsr(sa, sh);
            E_ADDRSR:     res = sa + f_rsr(sb, sh);
            E_MULRSR:     res = f_rsr(sa * sb, sh);
            E_MULRELURSR: res = f_rsr(sa * relu_b, sh);
            E_MULSUB:     res = sa * sb - sc;
            E_SRD:        res = f_srd(sa, sh, rnd);
            E_POW2Q:      res = f_pow2q(sa, sh[6:0], rnd);
            E_BITLENI:    res = {57'd0, f_bitlen(sa)} + simm;
            E_SEXPR:      res = sh;
            E_NRMS:       res = f_nrms(sa);
            E_NRMR:       res = f_rsr(rsq_prod, 64'sd16);
            E_EXP8:       res = {48'd0, exp8_lut(e8i)};
            E_EXP16:      res = e16ok ? {47'd0, exp16_lut(dd16[8:0])} : 64'sd0;
            E_OHSUB:      res = sa - (({32'd0, i0} == b) ? (64'sd1 <<< imm[5:0]) : 64'sd0);
            E_ONEHOT:     res = ({32'd0, i0} == a) ? 64'sd1 : 64'sd0;
            E_RELU2:      res = relu_a * relu_a;
            E_RSR2ADD:    res = f_rsr(sa, sr1) + f_rsr(sb, sr2);
            E_LIONQ:      res = sa - lion_u * (64'sd1 <<< r1[5:0]);
            E_MULADDI:    res = simm * sa + sb;
            E_RNG:        res = {32'd0, rnd};
            E_SHRF:       res = sa >>> (sh[63] ? 6'd0 : (sh > 63) ? 6'd63 : sh[5:0]);
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
    reg  [63:0] dv_q, dv_r, dv_d;
    reg  [6:0]  dv_n;
    reg         dv_busy, dv_neg;
    reg  [7:0]  dv_fn;
    wire [64:0] dv_rs = {dv_r, dv_q[63]};
    wire [64:0] dv_t = dv_rs - {1'b0, dv_d};
    wire signed [63:0] md_num = sa * 2 + simm;

    always @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            dv_busy <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && !is_div) begin
                y <= clampf(res);
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
                        dv_d <= simm * 2;
                    end
                    default: begin
                        dv_q <= a;
                        dv_d <= {32'd0, imm};
                    end
                endcase
                if (fn == E_RCP && (sa[63] || sa == 0)) begin
                    y <= clampf(64'sd0);
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
                    case (dv_fn)
                        E_MODU:    y <= clampf(dv_r);
                        E_MEANDIV: y <= clampf(dv_neg ? -$signed(dv_q) - ((dv_r != 0) ? 64'sd1 : 64'sd0) : dv_q);
                        default:   y <= clampf(dv_q);
                    endcase
                end
            end
        end
    end
endmodule
