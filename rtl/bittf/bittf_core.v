// BitTF jadro v1: provadi program ISA v1 (docs/bittf_isa.md) bitove shodne se simulatorem
// sw/bittf/isa/bt_isa.c. Verze v1 je sekvencni (jeden pristup do pameti naraz) - slouzi jako
// funkcni reference RTL; vykonova verze (pole MAC, vic drah, cache) ma stejne rozhrani.
//
// Pametovy port: pozadavek valid/ready (cteni nebo zapis prvku sirky 2^lw B), odpovedi na cteni
// v poradi pozadavku (data v dolnich bitech, bez rozsireni znamenka).
module bittf_core (
    input  wire        clk,
    input  wire        rst,
    // rizeni
    input  wire        start,
    input  wire [31:0] prog_base,
    output wire        busy,
    output reg         done,
    output reg  [7:0]  err,
    input  wire        csr_we,
    input  wire [1:0]  csr_addr,
    input  wire [31:0] csr_wdata,
    output reg  [31:0] csr_rdata,
    // pamet
    output reg         m_req_valid,
    input  wire        m_req_ready,
    output reg         m_req_we,
    output reg  [31:0] m_req_addr,
    output reg  [1:0]  m_req_lw,
    output reg  [63:0] m_req_wdata,
    input  wire        m_rsp_valid,
    input  wire [63:0] m_rsp_data,
    // ladeni
    output reg         dbg_instr_done,
    output reg  [31:0] dbg_pc,
    input  wire [5:0]  dbg_ridx,
    output wire [63:0] dbg_rdata
);
`include "bittf_isa.vh"
`include "bittf_func.vh"

    localparam ERR_FIT = 8'd1, ERR_MMOVF = 8'd2, ERR_OP = 8'd3;

    // ------------------------------------------------------------ stav
    localparam S_IDLE = 5'd0, S_FETCH = 5'd1, S_FWAIT = 5'd2, S_DEC = 5'd3, S_SCAL = 5'd4,
               S_SLD = 5'd5, S_LOOP = 5'd6, S_G = 5'd7, S_GW = 5'd8, S_RD = 5'd9, S_RDW = 5'd10,
               S_ALU = 5'd11, S_ALUW = 5'd12, S_WR = 5'd13, S_NEXT = 5'd14, S_END = 5'd15,
               S_ROWWR = 5'd16, S_HALT = 5'd17, S_ERR = 5'd18;
    reg [4:0] st;
    assign busy = st != S_IDLE;

    reg [31:0] ir [0:47];
    reg [31:0] pc, base;
    reg [63:0] R [0:63];
    reg [31:0] csr_seed, csr_ctr, csr_cyc;
    assign dbg_rdata = R[dbg_ridx];

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
    // operandy X = 0 D, 1 A, 2 B, 3 C
    function [31:0] obase(input integer x); obase = ir[5 + 6 * x]; endfunction
    function [1:0]  olw(input integer x);   olw = ir[6 + 6 * x][1:0]; endfunction
    function        ouns(input integer x);  ouns = ir[6 + 6 * x][2]; endfunction
    function        ogath(input integer x); ogath = ir[6 + 6 * x][3]; endfunction
    function [2:0]  omode(input integer x); omode = ir[6 + 6 * x][6:4]; endfunction
    function signed [63:0] ostr(input integer x, input integer k);
        ostr = {{32{ir[7 + 6 * x + k][31]}}, ir[7 + 6 * x + k]};
    endfunction
    wire [1:0] glw = ir[30][1:0];
    wire       guns = ir[30][2];
    wire signed [63:0] gscale = {{32{ir[35][31]}}, ir[35]};

    wire is_ew = op == OP_EW, is_red = op == OP_RED, is_mm = op == OP_MM;
    wire [2:0] tc = fl[2:0];
    wire [2:0] rpre = fl[10:8];
    wire need_a = is_mm || (is_ew ? EFN_NEED_A[fn[4:0]] : rpre != RP_SH);
    wire need_b = is_mm || (is_ew ? EFN_NEED_B[fn[4:0]] : (rpre == RP_AB || rpre == RP_ABC));
    wire need_c = is_ew ? (EFN_NEED_C[fn[4:0]] || tc != 0) : is_red && (rpre == RP_ABC || (rpre == RP_SH && tc != 0));
    wire uses_rng = is_ew && EFN_RNGMASK[fn[4:0]];

    // ------------------------------------------------------------ smycky a adresy
    // uroven 0 nejvnitrnejsi; MM: (k, i0, i1, i2)
    reg [31:0] cnt [0:3];
    reg [31:0] idx [0:3];
    reg [31:0] flat;
    reg signed [63:0] stv [0:4][0:3];   // kroky operandu D A B C a gatheru G po urovnich
    reg signed [63:0] off [0:4][0:3];   // offsety operandu na urovnich
    integer x, k;

    // uroven, ktera se zvysi (4 = konec)
    reg [2:0] adv;
    always @(*) begin
        if (idx[0] + 1 < cnt[0]) adv = 3'd0;
        else if (idx[1] + 1 < cnt[1]) adv = 3'd1;
        else if (idx[2] + 1 < cnt[2]) adv = 3'd2;
        else if (idx[3] + 1 < cnt[3]) adv = 3'd3;
        else adv = 3'd4;
    end

    // ------------------------------------------------------------ hodnoty prvku
    reg signed [63:0] va, vb, vc, gidx, acc;
    reg [2:0] rphase;           // ktery operand se cte: 1 A, 2 B, 3 C
    reg [31:0] key;
    wire [31:0] rnd = f_fmix32(flat * 32'h9E3779B1 + key);
    wire signed [63:0] gterm = gidx * gscale;

    function signed [63:0] ext(input [63:0] v, input [1:0] lw, input uns);
        case (lw)
            2'd0: ext = uns ? {56'd0, v[7:0]} : {{56{v[7]}}, v[7:0]};
            2'd1: ext = uns ? {48'd0, v[15:0]} : {{48{v[15]}}, v[15:0]};
            2'd2: ext = uns ? {32'd0, v[31:0]} : {{32{v[31]}}, v[31:0]};
            default: ext = v;
        endcase
    endfunction

    function fits(input signed [63:0] v, input [1:0] lw, input uns);
        case (lw)
            2'd0: fits = uns ? v[63:8] == 0 : (v[63:7] == 0 || &v[63:7]);
            2'd1: fits = uns ? v[63:16] == 0 : (v[63:15] == 0 || &v[63:15]);
            2'd2: fits = uns ? v[63:32] == 0 : (v[63:31] == 0 || &v[63:31]);
            default: fits = 1'b1;
        endcase
    endfunction

    function [31:0] eaddr(input integer xx, input signed [63:0] o, input signed [63:0] g);
        reg signed [63:0] e;
        begin
            e = o + (ogath(xx) ? g : 64'sd0);
            eaddr = obase(xx) + (e[31:0] << olw(xx));
        end
    endfunction

    // ------------------------------------------------------------ ALU
    reg alu_start;
    wire alu_done;
    wire [63:0] alu_y;
    bittf_alu alu (
        .clk(clk), .rst(rst), .start(alu_start), .fn(fn), .tc(tc), .clamp_en(fl[3] | fl[4]),
        .clamp_sym(fl[4]), .cbits(imm3[6:0]), .a(va), .b(vb), .c(vc), .r1(vR1), .r2(vR2), .imm(imm),
        .i0(idx[0]), .i1(idx[1]), .rnd(rnd), .done(alu_done), .y(alu_y)
    );

    // RED: predzpracovani a kombinace
    reg signed [63:0] rv, rtc, acc_n;
    always @(*) begin
        case (tc)
            TC_ADD: rtc = vc;
            TC_SUB: rtc = -vc;
            TC_ADDHALF: rtc = vc >>> 1;
            TC_SUBHALF: rtc = -(vc >>> 1);
            default: rtc = 64'sd0;
        endcase
        case (rpre)
            RP_AA:  rv = va * va;
            RP_AB:  rv = va * vb;
            RP_ABC: rv = va * vb * vc;
            RP_SH:  rv = simm + vR1 + rtc;
            default: rv = va;
        endcase
        case (fn)
            R_SUM:  acc_n = acc + rv;
            R_MAX:  acc_n = (rv > acc) ? rv : acc;
            default: acc_n = ((rv[63] ? -rv : rv) > acc) ? (rv[63] ? -rv : rv) : acc;
        endcase
    end
    wire signed [63:0] bsgn = (omode(2) == M_SIGN) ? (vb[63] ? -64'sd1 : 64'sd1) : vb;
    wire signed [63:0] mm_n = acc + va * bsgn;

    // vysledek EW k zapisu
    wire masked = fl[5] && idx[0] > idx[1];
    reg signed [63:0] wval;

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

    // ------------------------------------------------------------ hlavni automat
    reg [5:0] fcnt;            // nacitani instrukce: 24 slov po 64 bitech
    reg [5:0] frsp;

    task issue_read(input [31:0] ad, input [1:0] lw);
        begin
            m_req_valid <= 1'b1;
            m_req_we <= 1'b0;
            m_req_addr <= ad;
            m_req_lw <= lw;
        end
    endtask

    task issue_write(input [31:0] ad, input [1:0] lw, input [63:0] v);
        begin
            m_req_valid <= 1'b1;
            m_req_we <= 1'b1;
            m_req_addr <= ad;
            m_req_lw <= lw;
            m_req_wdata <= v;
        end
    endtask

    // pocatecni nastaveni smycek pro instrukci
    task loop_init;
        integer xx, kk;
        begin
            for (xx = 0; xx < 5; xx = xx + 1)
                for (kk = 0; kk < 4; kk = kk + 1)
                    off[xx][kk] <= 64'sd0;
            for (kk = 0; kk < 4; kk = kk + 1)
                idx[kk] <= 32'd0;
            flat <= 32'd0;
            if (is_mm) begin
                cnt[0] <= ir[4]; cnt[1] <= ir[1]; cnt[2] <= ir[2]; cnt[3] <= ir[3];
                // D: (0, s0, s1, s2); A: (s3, 0, s1, s2); B: (s3, s0, 0, s2)
                stv[0][0] <= 0;          stv[0][1] <= ostr(0, 0); stv[0][2] <= ostr(0, 1); stv[0][3] <= ostr(0, 2);
                stv[1][0] <= ostr(1, 3); stv[1][1] <= 0;          stv[1][2] <= ostr(1, 1); stv[1][3] <= ostr(1, 2);
                stv[2][0] <= ostr(2, 3); stv[2][1] <= ostr(2, 0); stv[2][2] <= 0;          stv[2][3] <= ostr(2, 2);
                for (kk = 0; kk < 4; kk = kk + 1) begin
                    stv[3][kk] <= 0;
                    stv[4][kk] <= 0;
                end
            end else begin
                for (kk = 0; kk < 4; kk = kk + 1) begin
                    cnt[kk] <= ir[1 + kk];
                    for (xx = 0; xx < 4; xx = xx + 1)
                        stv[xx][kk] <= ostr(xx, kk);
                    stv[4][kk] <= {{32{ir[31 + kk][31]}}, ir[31 + kk]};
                end
                if (is_red)
                    stv[0][0] <= 0;
            end
        end
    endtask

    // posun smycek o jeden prvek na urovni adv
    task loop_adv;
        integer xx, kk;
        begin
            for (kk = 0; kk < 4; kk = kk + 1) begin
                if (kk < adv) idx[kk] <= 32'd0;
                else if (kk == adv) idx[kk] <= idx[kk] + 32'd1;
            end
            for (xx = 0; xx < 5; xx = xx + 1)
                for (kk = 0; kk < 4; kk = kk + 1)
                    if (kk <= adv && adv < 4)
                        off[xx][kk] <= off[xx][adv] + stv[xx][adv];
            flat <= flat + 32'd1;
        end
    endtask

    wire any_gath = ogath(0) | ogath(1) | ogath(2) | ogath(3);
    reg signed [63:0] gidx_r;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE;
            done <= 1'b0;
            err <= 8'd0;
            m_req_valid <= 1'b0;
            alu_start <= 1'b0;
            dbg_instr_done <= 1'b0;
            csr_seed <= 32'd0;
            csr_ctr <= 32'd0;
            csr_cyc <= 32'd0;
            for (k = 0; k < 64; k = k + 1)
                R[k] <= 64'd0;
        end else begin
            done <= 1'b0;
            dbg_instr_done <= 1'b0;
            alu_start <= 1'b0;
            if (st != S_IDLE)
                csr_cyc <= csr_cyc + 32'd1;
            if (m_req_valid && m_req_ready)
                m_req_valid <= 1'b0;
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
                        pc <= 32'd0;
                        err <= 8'd0;
                        st <= S_FETCH;
                        fcnt <= 6'd0;
                        frsp <= 6'd0;
                    end
                // --- nacteni instrukce (24 cteni po 8 B, pipelinovane)
                S_FETCH: begin
                    if (m_rsp_valid) begin
                        ir[2 * frsp] <= m_rsp_data[31:0];
                        ir[2 * frsp + 1] <= m_rsp_data[63:32];
                        frsp <= frsp + 6'd1;
                    end
                    if ((!m_req_valid || m_req_ready) && fcnt < 24) begin
                        issue_read(base + pc * 192 + {26'd0, fcnt} * 8, 2'd3);
                        fcnt <= fcnt + 6'd1;
                    end
                    if (frsp == 23 && m_rsp_valid)
                        st <= S_DEC;
                end
                S_DEC: begin
                    dbg_pc <= pc;
                    case (op)
                        OP_NOP: st <= S_END;
                        OP_HALT: st <= S_HALT;
                        OP_SCALAR: st <= S_SCAL;
                        OP_BRANCH: begin
                            if ((fn == B_BEQZ && vR1 == 0) || (fn == B_BNEZ && vR1 != 0) || fn == B_JMP)
                                pc <= imm - 32'd1;   // S_END pricte 1
                            st <= S_END;
                        end
                        OP_EW, OP_RED, OP_MM: begin
                            loop_init;
                            acc <= (is_red && fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                            if (uses_rng) begin
                                key <= f_fmix32(csr_seed ^ f_fmix32(csr_ctr));
                                csr_ctr <= csr_ctr + 32'd1;
                            end
                            st <= S_LOOP;
                        end
                        default: begin err <= ERR_OP; st <= S_ERR; end
                    endcase
                end
                // --- skalarni instrukce
                S_SCAL: begin
                    if (fn == S_LD) begin
                        issue_read(obase(0), olw(0));
                        st <= S_SLD;
                    end else if (fn == S_ST) begin
                        if (!fits(vR1, olw(0), ouns(0))) begin
                            err <= ERR_FIT; st <= S_ERR;
                        end else begin
                            issue_write(obase(0), olw(0), vR1);
                            st <= S_END;
                        end
                    end else begin
                        if (fn == S_CSRW) begin
                            case (imm[1:0])
                                2'd0: csr_seed <= vR1[31:0];
                                2'd1: csr_ctr <= vR1[31:0];
                                2'd2: err <= vR1[7:0];
                                default: csr_cyc <= vR1[31:0];
                            endcase
                        end else if (swr && rd != 0)
                            R[rd] <= sres;
                        else if (!swr && fn != S_CSRW) begin
                            err <= ERR_OP; st <= S_ERR;
                        end
                        if (swr || fn == S_CSRW)
                            st <= S_END;
                    end
                end
                S_SLD:
                    if (m_rsp_valid) begin
                        if (rd != 0) R[rd] <= ext(m_rsp_data, olw(0), ouns(0));
                        st <= S_END;
                    end
                // --- smycka prvku (EW, RED, MM)
                S_LOOP: begin
                    if (any_gath && !is_mm) begin
                        issue_read(base_g(off[4][0]), glw);
                        st <= S_GW;
                    end else begin
                        gidx_r <= 0;
                        gidx <= 0;
                        rphase <= 3'd1;
                        st <= S_RD;
                    end
                end
                S_GW:
                    if (m_rsp_valid) begin
                        gidx <= ext(m_rsp_data, glw, guns);
                        rphase <= 3'd1;
                        st <= S_RD;
                    end
                S_RD: begin
                    // dalsi potrebny operand podle rphase
                    if (rphase == 1 && need_a) begin
                        issue_read(eaddr(1, off[1][0], gterm), olw(1));
                        st <= S_RDW;
                    end else if (rphase <= 2 && need_b) begin
                        rphase <= 3'd2;
                        issue_read(eaddr(2, off[2][0], gterm), olw(2));
                        st <= S_RDW;
                    end else if (rphase <= 3 && need_c) begin
                        rphase <= 3'd3;
                        issue_read(eaddr(3, off[3][0], gterm), olw(3));
                        st <= S_RDW;
                    end else
                        st <= S_ALU;
                end
                S_RDW:
                    if (m_rsp_valid) begin
                        case (rphase)
                            3'd1: va <= ext(m_rsp_data, olw(1), ouns(1));
                            3'd2: vb <= ext(m_rsp_data, olw(2), ouns(2));
                            default: vc <= ext(m_rsp_data, olw(3), ouns(3));
                        endcase
                        rphase <= rphase + 3'd1;
                        st <= S_RD;
                    end
                S_ALU: begin
                    if (is_ew) begin
                        if (masked) begin
                            wval <= vR2;
                            st <= S_WR;
                        end else begin
                            alu_start <= 1'b1;
                            st <= S_ALUW;
                        end
                    end else if (is_red) begin
                        acc <= acc_n;
                        st <= (adv == 0) ? S_NEXT : S_ROWWR;
                    end else begin
                        acc <= mm_n;
                        st <= (adv == 0) ? S_NEXT : S_ROWWR;
                    end
                end
                S_ALUW:
                    if (alu_done) begin
                        wval <= alu_y;
                        st <= S_WR;
                    end
                S_WR: begin
                    if (!fits(wval, olw(0), ouns(0))) begin
                        err <= ERR_FIT; st <= S_ERR;
                    end else begin
                        issue_write(eaddr(0, off[0][0], gterm), olw(0), wval);
                        st <= S_NEXT;
                    end
                end
                // konec radku RED / vystupu MM
                S_ROWWR: begin
                    if (is_mm && (acc >= 64'sh0000800000000000 || acc < -64'sh0000800000000000)) begin
                        err <= ERR_MMOVF; st <= S_ERR;
                    end else if (is_red && fl[11]) begin
                        if (rd != 0) R[rd] <= acc;
                        acc <= (fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                        st <= S_NEXT;
                    end else if (!fits(acc, olw(0), ouns(0))) begin
                        err <= ERR_FIT; st <= S_ERR;
                    end else if (!m_req_valid || m_req_ready) begin
                        issue_write(eaddr(0, off[0][0], 64'sd0), olw(0), acc);
                        acc <= (is_red && fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                        st <= S_NEXT;
                    end
                end
                S_NEXT:
                    if (!m_req_valid || m_req_ready) begin
                        if (adv == 4)
                            st <= S_END;
                        else begin
                            loop_adv;
                            st <= S_LOOP;
                        end
                    end
                S_END:
                    if (!m_req_valid || m_req_ready) begin
                        dbg_instr_done <= 1'b1;
                        pc <= pc + 32'd1;
                        fcnt <= 6'd0;
                        frsp <= 6'd0;
                        st <= S_FETCH;
                    end
                S_HALT: begin
                    done <= 1'b1;
                    dbg_instr_done <= 1'b1;
                    st <= S_IDLE;
                end
                S_ERR: begin
                    done <= 1'b1;
                    st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end

    function [31:0] base_g(input signed [63:0] o);
        base_g = ir[29] + (o[31:0] << glw);
    endfunction
endmodule
