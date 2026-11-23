// BitTF mini: jadro v1 (stejna ISA, bitove shodne s sw/bittf/isa/bt_isa.c) zmensene pro maly cip
// (bezplatna plocha akademickeho MPW, IHP SG13G2 130 nm, ~2 mm^2 vcetne padu):
//  - registry R v jednoportove SRAM (bittf_rf -> makro 64x64), cteni rs1/rs2/rs3 po nacteni instrukce,
//  - zadne kombinacni nasobicky: jedna postupna bittf_mulseq pro ALU, RED, MM, gather i fmix32
//    (nahodna cisla se pocitaji postupne jen pro instrukce, ktere je pouzivaji),
//  - kroky adres se ctou primo z instrukce, offsety 32 bitu (adresa ma 32 bitu).
//
// Pametovy port: pozadavek valid/ready (cteni nebo zapis prvku sirky 2^lw B), odpovedi na cteni
// v poradi pozadavku (data v dolnich bitech, bez rozsireni znamenka).
module bittf_mini (
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
               S_ROWWR = 5'd16, S_HALT = 5'd17, S_ERR = 5'd18, S_RR = 5'd19, S_GM = 5'd20,
               S_MULW = 5'd21, S_ACC = 5'd22, S_CLR = 5'd23, S_AMW = 5'd24, S_FMS = 5'd25,
               S_FMW1 = 5'd26, S_FMW2 = 5'd27;
    reg [4:0] st;
    assign busy = st != S_IDLE;

    reg [31:0] ir [0:47];
    reg [31:0] pc, base;
    // registrova sada v SRAM: zapis odlozeny o takt (wr_*), cteni jen ve stavu S_RR
    reg        wr_pend;
    reg [5:0]  wr_addr, rrc;
    reg [63:0] wr_data;
    reg signed [63:0] rq1, rq2, rq3;
    wire        rf_rd = st == S_RR && rrc < 3 && !wr_pend;
    wire        rf_en = rf_rd || wr_pend || st == S_CLR;
    wire        rf_we = wr_pend || st == S_CLR;
    wire [5:0]  rf_addr = st == S_CLR ? rrc : wr_pend ? wr_addr :
                          rrc == 0 ? ir[36][5:0] : rrc == 1 ? ir[36][13:8] : ir[36][29:24];
    wire [63:0] rf_q;
    wire [63:0] rf_dbg;
    bittf_rf rf (.clk(clk), .en(rf_en), .we(rf_we), .addr(rf_addr), .wdata(st == S_CLR ? 64'd0 : wr_data),
                 .rdata(rf_q), .dbg_addr(dbg_ridx), .dbg_rdata(rf_dbg));
    reg [31:0] csr_seed, csr_ctr, csr_cyc;
    assign dbg_rdata = rf_dbg;

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
    wire signed [63:0] vR1 = rq1, vR2 = rq2, vR3 = rq3;
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
    reg [31:0] flat, fh;                // fh = flat * 0x9E3779B1 (prubezne)
    // krok operandu x (0..3 = D A B C, 4 = gather G) na urovni k, primo z instrukce
    function [31:0] stvf(input integer x, input integer k);
        begin
            if (is_mm) begin
                // D: (0, s0, s1, s2); A: (s3, 0, s1, s2); B: (s3, s0, 0, s2)
                if (x == 0) stvf = (k == 0) ? 32'd0 : ir[7 + 6 * x + k - 1];
                else if (x == 1) stvf = (k == 0) ? ir[7 + 6 + 3] : (k == 1) ? 32'd0 : ir[7 + 6 + k - 1];
                else if (x == 2) stvf = (k == 0) ? ir[7 + 12 + 3] : (k == 1) ? ir[7 + 12] : (k == 2) ? 32'd0 : ir[7 + 12 + 2];
                else stvf = 32'd0;
            end else if (x == 4)
                stvf = ir[31 + k];
            else if (x == 0 && k == 0 && is_red)
                stvf = 32'd0;
            else
                stvf = ir[7 + 6 * x + k];
        end
    endfunction
    function [31:0] stva(input integer x, input [2:0] a);
        case (a)
            3'd0: stva = stvf(x, 0);
            3'd1: stva = stvf(x, 1);
            3'd2: stva = stvf(x, 2);
            default: stva = stvf(x, 3);
        endcase
    endfunction
    reg [31:0] off [0:4][0:3];          // offsety operandu na urovnich
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
    reg signed [63:0] va, vb, vc, acc;
    reg [2:0] rphase;           // ktery operand se cte: 1 A, 2 B, 3 C
    reg [31:0] key;
    reg  [31:0] rnd;                    // fmix32(flat * 0x9E3779B1 + key), spocteno pred ALU
    reg  signed [63:0] gterm;           // gidx * gscale (z postupne nasobicky)

    // jedna postupna nasobicka; start a operandy kombinacne podle stavu (viz nize)
    reg         m_go;
    reg  [63:0] m_a, m_b;
    reg  [1:0]  mphase;
    wire        m_done;
    wire [63:0] m_p;
    wire signed [63:0] prd = m_p;       // plati od m_done do dalsiho startu
    bittf_mulseq #(.W(64), .D(8)) mul (.clk(clk), .rst(rst), .start(m_go), .a(m_a), .b(m_b), .done(m_done), .p(m_p));
    wire        alu_mul;
    wire [63:0] alu_ma, alu_mb;

    // fmix32 po krocich: h1 = h^(h>>16); h2 = h1*C1; h3 = h2^(h2>>13); h4 = h3*C2; vysledek h4^(h4>>16)
    reg  [31:0] fm_in;
    reg  [1:0]  fm_ret;                 // 0 = klic (1. kolo), 1 = klic (2. kolo), 2 = rnd prvku
    localparam FM_KEY1 = 2'd0, FM_KEY2 = 2'd1, FM_RND = 2'd2;
    wire [31:0] fm_h1 = fm_in ^ (fm_in >> 16);
    wire [31:0] fm_h3 = m_p[31:0] ^ (m_p[31:0] >> 13);
    wire [31:0] fm_out = m_p[31:0] ^ (m_p[31:0] >> 16);

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

    // jedna adresni jednotka: operand au_x (0..3 = D A B C, 4 = index gatheru), au_g = gather k offsetu
    reg  [2:0]  au_x;
    reg         au_gz;                  // gather vynulovat (zapis radku RED/MM)
    reg  [31:0] au_base, au_off;
    reg  [1:0]  au_lw;
    reg         au_gath;
    always @(*) begin
        if (st == S_LOOP) au_x = 3'd4;
        else if (st == S_RD) au_x = (rphase == 1 && need_a) ? 3'd1 : (rphase <= 2 && need_b) ? 3'd2 : 3'd3;
        else au_x = 3'd0;
        au_gz = st == S_ROWWR;
        case (au_x)
            3'd0: begin au_base = obase(0); au_off = off[0][0]; au_lw = olw(0); au_gath = ogath(0); end
            3'd1: begin au_base = obase(1); au_off = off[1][0]; au_lw = olw(1); au_gath = ogath(1); end
            3'd2: begin au_base = obase(2); au_off = off[2][0]; au_lw = olw(2); au_gath = ogath(2); end
            3'd3: begin au_base = obase(3); au_off = off[3][0]; au_lw = olw(3); au_gath = ogath(3); end
            default: begin au_base = ir[29]; au_off = off[4][0]; au_lw = glw; au_gath = 1'b0; end
        endcase
    end
    wire [31:0] au_e = au_off + ((au_gath && !au_gz) ? gterm[31:0] : 32'd0);
    wire [31:0] au_addr = au_base + (au_e << au_lw);

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
    bittf_mini_alu alu (
        .clk(clk), .rst(rst), .start(alu_start), .fn(fn), .tc(tc), .clamp_en(fl[3] | fl[4]),
        .clamp_sym(fl[4]), .cbits(imm3[6:0]), .a(va), .b(vb), .c(vc), .r1(vR1), .r2(vR2), .imm(imm),
        .i0(idx[0]), .i1(idx[1]), .rnd(rnd), .mul_need(alu_mul), .mul_a(alu_ma), .mul_b(alu_mb),
        .mul_p(m_p), .done(alu_done), .y(alu_y)
    );

    // RED: predzpracovani a kombinace
    reg signed [63:0] rv, rtc, acc_n;
    wire signed [63:0] rv_c = (fn == R_MAX || !rv[63]) ? rv : -rv;
    always @(*) begin
        case (tc)
            TC_ADD: rtc = vc;
            TC_SUB: rtc = -vc;
            TC_ADDHALF: rtc = vc >>> 1;
            TC_SUBHALF: rtc = -(vc >>> 1);
            default: rtc = 64'sd0;
        endcase
        case (rpre)
            RP_AA, RP_AB, RP_ABC: rv = prd;
            RP_SH:  rv = simm + vR1 + rtc;
            default: rv = va;
        endcase
        case (fn)
            R_SUM:  acc_n = acc + rv;
            default: acc_n = (rv_c > acc) ? rv_c : acc;   // MAX: rv, ABSMAX: |rv| (jeden komparator)
        endcase
    end
    wire mm_sign = omode(2) == M_SIGN;
    wire signed [63:0] mm_n = acc + (mm_sign ? (vb[63] ? -va : va) : prd);
    wire red_mul = rpre == RP_AA || rpre == RP_AB || rpre == RP_ABC;

    // vysledek EW k zapisu
    wire masked = fl[5] && idx[0] > idx[1];
    reg signed [63:0] wval;

    // ------------------------------------------------------------ skalarni jednotka
    reg signed [63:0] sres;
    // spolecna scitacka a komparator skalarni jednotky
    wire signed [63:0] s_sum = vR1 + (fn == S_SUB ? -vR2 : vR2) + simm;
    wire s_gt = vR1 > vR2;
    wire s_lt = !s_gt && vR1 != vR2;
    reg swr;
    always @(*) begin
        swr = 1'b1;
        case (fn)
            S_LI:     sres = simm + (simm2 <<< 32);
            S_ADD,
            S_SUB:    sres = s_sum;
            S_MAX:    sres = s_gt ? vR1 : vR2;
            S_MIN:    sres = s_lt ? vR1 : vR2;
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
                    off[xx][kk] <= 32'd0;
            for (kk = 0; kk < 4; kk = kk + 1)
                idx[kk] <= 32'd0;
            flat <= 32'd0;
            fh <= 32'd0;
            if (is_mm) begin
                cnt[0] <= ir[4]; cnt[1] <= ir[1]; cnt[2] <= ir[2]; cnt[3] <= ir[3];
            end else begin
                for (kk = 0; kk < 4; kk = kk + 1)
                    cnt[kk] <= ir[1 + kk];
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
                        off[xx][kk] <= off[xx][adv] + stva(xx, adv);
            flat <= flat + 32'd1;
            fh <= fh + 32'h9E3779B1;
        end
    endtask

    wire any_gath = ogath(0) | ogath(1) | ogath(2) | ogath(3);

    // start nasobicky: ve stejnem taktu, kdy automat prechazi do cekaciho stavu
    always @(*) begin
        m_go = 1'b0;
        m_a = 64'd0;
        m_b = 64'd0;
        case (st)
            S_GW: if (m_rsp_valid) begin m_go = 1'b1; m_a = ext(m_rsp_data, glw, guns); m_b = gscale; end
            S_ALU:
                if (is_ew) begin
                    if (!masked && alu_mul) begin m_go = 1'b1; m_a = alu_ma; m_b = alu_mb; end
                end else if ((is_red && red_mul) || (is_mm && !mm_sign)) begin
                    m_go = 1'b1; m_a = va; m_b = (is_red && rpre == RP_AA) ? va : vb;
                end
            S_MULW: if (m_done && mphase == 1 && is_red && rpre == RP_ABC) begin m_go = 1'b1; m_a = m_p; m_b = vc; end
            S_FMS: begin m_go = 1'b1; m_a = {32'd0, fm_h1}; m_b = 64'h85EBCA6B; end
            S_FMW1: if (m_done) begin m_go = 1'b1; m_a = {32'd0, fm_h3}; m_b = 64'hC2B2AE35; end
            default: ;
        endcase
    end
    reg start_pend;

    always @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            err <= 8'd0;
            m_req_valid <= 1'b0;
            alu_start <= 1'b0;
            dbg_instr_done <= 1'b0;
            csr_seed <= 32'd0;
            csr_ctr <= 32'd0;
            csr_cyc <= 32'd0;
            st <= S_CLR;
            rrc <= 6'd0;
            wr_pend <= 1'b0;
            start_pend <= 1'b0;
        end else begin
            if (wr_pend) wr_pend <= 1'b0;
            if (start) start_pend <= 1'b1;
            done <= 1'b0;
            dbg_instr_done <= 1'b0;
            alu_start <= 1'b0;
            if (st != S_IDLE)
                csr_cyc <= csr_cyc + 32'd1;
            if (m_req_valid && m_req_ready)
                m_req_valid <= 1'b0;
            if (csr_we && (st == S_IDLE || st == S_CLR)) begin
                case (csr_addr)
                    2'd0: csr_seed <= csr_wdata;
                    2'd1: csr_ctr <= csr_wdata;
                    2'd2: err <= csr_wdata[7:0];
                    default: csr_cyc <= csr_wdata;
                endcase
            end
            case (st)
                S_CLR: begin
                    rrc <= rrc + 6'd1;
                    if (rrc == 6'd63) st <= S_IDLE;
                end
                S_IDLE:
                    if (start || start_pend) begin
                        start_pend <= 1'b0;
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
                    if (frsp == 23 && m_rsp_valid) begin
                        rrc <= 6'd0;
                        st <= S_RR;
                    end
                end
                // --- cteni rs1, rs2, rs3 ze SRAM (data takt po adrese)
                S_RR:
                    if (!wr_pend) begin
                        rrc <= rrc + 6'd1;
                        if (rrc == 1) rq1 <= rf_q;
                        if (rrc == 2) rq2 <= rf_q;
                        if (rrc == 3) begin rq3 <= rf_q; st <= S_DEC; end
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
                                // klic = fmix32(seed ^ fmix32(ctr)), postupne pres nasobicku
                                fm_in <= csr_ctr;
                                fm_ret <= FM_KEY1;
                                csr_ctr <= csr_ctr + 32'd1;
                                st <= S_FMS;
                            end else
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
                            rf_write(rd, sres);
                        else if (!swr && fn != S_CSRW) begin
                            err <= ERR_OP; st <= S_ERR;
                        end
                        if (swr || fn == S_CSRW)
                            st <= S_END;
                    end
                end
                S_SLD:
                    if (m_rsp_valid) begin
                        if (rd != 0) rf_write(rd, ext(m_rsp_data, olw(0), ouns(0)));
                        st <= S_END;
                    end
                // --- smycka prvku (EW, RED, MM)
                S_LOOP: begin
                    if (any_gath && !is_mm) begin
                        issue_read(au_addr, glw);
                        st <= S_GW;
                    end else begin
                        gterm <= 0;
                        rphase <= 3'd1;
                        st <= S_RD;
                    end
                end
                S_GW:
                    if (m_rsp_valid) begin
                        st <= S_GM;   // nasobicka startuje kombinacne (m_go)
                    end
                S_GM:
                    if (m_done) begin
                        gterm <= m_p;
                        rphase <= 3'd1;
                        st <= S_RD;
                    end
                S_RD: begin
                    // dalsi potrebny operand podle rphase
                    if (rphase == 1 && need_a) begin
                        issue_read(au_addr, olw(1));
                        st <= S_RDW;
                    end else if (rphase <= 2 && need_b) begin
                        rphase <= 3'd2;
                        issue_read(au_addr, olw(2));
                        st <= S_RDW;
                    end else if (rphase <= 3 && need_c) begin
                        rphase <= 3'd3;
                        issue_read(au_addr, olw(3));
                        st <= S_RDW;
                    end else
                        st <= S_ALU;
                    if (!(rphase == 1 && need_a) && !(rphase <= 2 && need_b) && !(rphase <= 3 && need_c) && uses_rng) begin
                        fm_in <= fh + key;
                        fm_ret <= FM_RND;
                        st <= S_FMS;
                    end
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
                        end else if (alu_mul) begin
                            st <= S_AMW;          // nejdriv soucin pro ALU
                        end else begin
                            alu_start <= 1'b1;
                            st <= S_ALUW;
                        end
                    end else if ((is_red && red_mul) || (is_mm && !mm_sign)) begin
                        mphase <= 2'd1;
                        st <= S_MULW;
                    end else begin
                        acc <= is_red ? acc_n : mm_n;
                        st <= (adv == 0) ? S_NEXT : S_ROWWR;
                    end
                end
                S_MULW:
                    if (m_done) begin
                        if (mphase == 1 && is_red && rpre == RP_ABC)
                            mphase <= 2'd2;       // druhy soucin (*vc) startuje kombinacne
                        else
                            st <= S_ACC;
                    end
                S_ACC: begin
                    acc <= is_red ? acc_n : mm_n;
                    st <= (adv == 0) ? S_NEXT : S_ROWWR;
                end
                S_AMW:
                    if (m_done) begin
                        alu_start <= 1'b1;
                        st <= S_ALUW;
                    end
                // --- fmix32 pres nasobicku
                S_FMS: st <= S_FMW1;              // start s h1*C1
                S_FMW1: if (m_done) st <= S_FMW2; // start s h3*C2
                S_FMW2:
                    if (m_done) begin
                        case (fm_ret)
                            FM_KEY1: begin fm_in <= csr_seed ^ fm_out; fm_ret <= FM_KEY2; st <= S_FMS; end
                            FM_KEY2: begin key <= fm_out; st <= S_LOOP; end
                            default: begin rnd <= fm_out; st <= S_ALU; end
                        endcase
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
                        issue_write(au_addr, olw(0), wval);
                        st <= S_NEXT;
                    end
                end
                // konec radku RED / vystupu MM
                S_ROWWR: begin
                    if (is_mm && (acc >= 64'sh0000800000000000 || acc < -64'sh0000800000000000)) begin
                        err <= ERR_MMOVF; st <= S_ERR;
                    end else if (is_red && fl[11]) begin
                        if (rd != 0) rf_write(rd, acc);
                        acc <= (fn == R_MAX) ? 64'sh8000000000000000 : 64'sd0;
                        st <= S_NEXT;
                    end else if (!fits(acc, olw(0), ouns(0))) begin
                        err <= ERR_FIT; st <= S_ERR;
                    end else if (!m_req_valid || m_req_ready) begin
                        issue_write(au_addr, olw(0), acc);
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

    // zapis do registrove sady (provede se v pristim taktu) + aktualizace nactenych kopii
    task rf_write(input [5:0] a, input [63:0] v);
        begin
            wr_pend <= 1'b1;
            wr_addr <= a;
            wr_data <= v;
            if (a == rs1) rq1 <= v;
            if (a == rs2) rq2 <= v;
            if (a == rs3) rq3 <= v;
        end
    endtask

    function [31:0] base_g(input signed [63:0] o);
        base_g = ir[29] + (o[31:0] << glw);
    endfunction
endmodule
