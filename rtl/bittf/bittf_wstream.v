// BitTF v2: proudovy zapis vysledku. Prochazi stejnou 4-D smycku jako bittf_stream, takty (beat) az VL
// prvku urovne 0 (krok s0 = 1) nebo po 1 prvku. Sklada zapisy do celych radku s bajtovou maskou;
// radek odesle, kdyz takt zacne jinym radkem nebo radek prekroci. busy drzi, dokud nejsou potvrzeny
// vsechny zapisy (instrukce smi skoncit az potom).
module bittf_wstream #(
    parameter VL = 8,
    parameter LB = 64
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire [31:0]       base,
    input  wire [1:0]        lw,
    input  wire              vec,
    input  wire [127:0]      cnt,
    input  wire [255:0]      str,
    input  wire              flush,      // konec instrukce: odeslat rozpracovany radek
    // takty dat
    input  wire              i_valid,
    output wire              i_ready,
    input  wire [VL*64-1:0]  i_data,
    // radkovy port (zapis)
    output reg               wq_valid,
    input  wire              wq_ready,
    output reg  [25:0]       wq_line,
    output reg  [LB*8-1:0]   wq_data,
    output reg  [LB-1:0]     wq_strb,
    input  wire              wack,
    output wire              busy
);
    localparam LBB = $clog2(LB);

    reg [31:0] c [0:3];
    reg [31:0] i [0:3];
    reg signed [63:0] s [0:3];
    reg signed [63:0] off [0:3];
    integer k, l;

    wire [31:0] rem0 = c[0] - i[0];
    wire [31:0] vmax = ((LB >> lw) < VL) ? (LB >> lw) : VL;
    wire [6:0]  blen = (!vec) ? 7'd1 : (rem0 >= vmax) ? vmax[6:0] : rem0[6:0];
    wire [31:0] baddr = base + (off[0][31:0] << lw);
    wire [31:0] bytes = {25'd0, blen} << lw;
    wire [31:0] alast = baddr + bytes - 32'd1;
    wire [25:0] L0 = baddr[31:LBB], L1 = alast[31:LBB];
    wire span = L1 != L0;

    reg [2:0] adv;
    always @(*) begin
        if (i[0] + blen < c[0]) adv = 3'd0;
        else if (i[1] + 1 < c[1]) adv = 3'd1;
        else if (i[2] + 1 < c[2]) adv = 3'd2;
        else if (i[3] + 1 < c[3]) adv = 3'd3;
        else adv = 3'd4;
    end

    // zabaleni prvku do bajtu
    reg [VL*64-1:0] pk;
    reg [VL*8-1:0]  pm;
    always @(*) begin
        pk = 0;
        pm = 0;
        for (l = 0; l < VL; l = l + 1) begin
            if (l < blen) begin
                case (lw)
                    2'd0: begin pk[8 * l +: 8] = i_data[64 * l +: 8]; pm[l] = 1'b1; end
                    2'd1: begin pk[16 * l +: 16] = i_data[64 * l +: 16]; pm[2 * l +: 2] = 2'b11; end
                    2'd2: begin pk[32 * l +: 32] = i_data[64 * l +: 32]; pm[4 * l +: 4] = 4'hF; end
                    default: begin pk[64 * l +: 64] = i_data[64 * l +: 64]; pm[8 * l +: 8] = 8'hFF; end
                endcase
            end
        end
    end
    wire [2*LB*8-1:0] wd = {{(2*LB*8-VL*64){1'b0}}, pk} << {baddr[LBB-1:0], 3'b000};
    wire [2*LB-1:0]   wm = {{(2*LB-VL*8){1'b0}}, pm} << baddr[LBB-1:0];

    // rozpracovany radek
    reg            bv;
    reg [25:0]     bl;
    reg [LB*8-1:0] bd;
    reg [LB-1:0]   bm;

    // fronta zapisu (az 2 radky na takt)
    reg [26+LB*8+LB-1:0] fq [0:3];
    reg [2:0] fq_n;
    reg [1:0] fq_w, fq_r;
    reg [15:0] outst;            // nepotvrzene zapisy

    // kolik radku takt odesle: stary (jiny nez L0), L0 pri prekroceni
    wire flush_old = bv && bl != L0;
    wire [1:0] nout = {1'b0, flush_old} + {1'b0, span};
    assign i_ready = fq_n + nout <= 4 && !start;
    wire take = i_valid && i_ready;

    // sloucene L0 (se starym radkem, pokud je stejny)
    function [LB*8-1:0] merge(input [LB*8-1:0] od, input [LB-1:0] om, input [LB*8-1:0] nd, input [LB-1:0] nm);
        integer b;
        begin
            for (b = 0; b < LB; b = b + 1)
                merge[8 * b +: 8] = nm[b] ? nd[8 * b +: 8] : od[8 * b +: 8];
        end
    endfunction
    wire [LB*8-1:0] m0d = (bv && bl == L0) ? merge(bd, bm, wd[LB*8-1:0], wm[LB-1:0]) : wd[LB*8-1:0];
    wire [LB-1:0]   m0m = (bv && bl == L0) ? (bm | wm[LB-1:0]) : wm[LB-1:0];

    wire do_flush = flush && bv && !take && fq_n < 4;
    assign busy = bv || fq_n != 0 || wq_valid || outst != 0;

    always @(posedge clk) begin
        if (rst) begin
            bv <= 1'b0;
            fq_n <= 0;
            fq_w <= 0;
            fq_r <= 0;
            wq_valid <= 1'b0;
            outst <= 0;
        end else begin
            // vystup fronty na port
            if (wq_valid && wq_ready) wq_valid <= 1'b0;
            if ((!wq_valid || wq_ready) && fq_n != 0) begin
                {wq_line, wq_data, wq_strb} <= fq[fq_r];
                wq_valid <= 1'b1;
                fq_r <= fq_r + 1'b1;
            end
            outst <= outst + ((wq_valid && wq_ready) ? 1'b1 : 1'b0) - (wack ? 1'b1 : 1'b0);
            if (start) begin
                for (k = 0; k < 4; k = k + 1) begin
                    c[k] <= cnt[32 * k +: 32];
                    s[k] <= str[64 * k +: 64];
                    i[k] <= 32'd0;
                    off[k] <= 64'sd0;
                end
            end else if (take) begin
                case ({flush_old, span})
                    2'b10: begin
                        fq[fq_w] <= {bl, bd, bm};
                        fq_w <= fq_w + 2'd1;
                    end
                    2'b01: begin
                        fq[fq_w] <= {L0, m0d, m0m};
                        fq_w <= fq_w + 2'd1;
                    end
                    2'b11: begin
                        fq[fq_w] <= {bl, bd, bm};
                        fq[fq_w + 2'd1] <= {L0, m0d, m0m};
                        fq_w <= fq_w + 2'd2;
                    end
                    default: ;
                endcase
                bv <= 1'b1;
                if (span) begin
                    bl <= L1;
                    bd <= wd[2*LB*8-1:LB*8];
                    bm <= wm[2*LB-1:LB];
                end else begin
                    bl <= L0;
                    bd <= m0d;
                    bm <= m0m;
                end
                for (k = 0; k < 4; k = k + 1) begin
                    if (k < adv) i[k] <= 32'd0;
                    else if (k == adv) i[k] <= i[k] + ((k == 0) ? {25'd0, blen} : 32'd1);
                end
                for (k = 0; k < 4; k = k + 1)
                    if (k <= adv && adv < 4)
                        off[k] <= off[adv] + ((adv == 0) ? (vec ? {57'd0, blen} : s[0]) : s[adv]);
            end else if (do_flush) begin
                fq[fq_w] <= {bl, bd, bm};
                fq_w <= fq_w + 2'd1;
                bv <= 1'b0;
            end
            fq_n <= fq_n + (take ? {1'b0, nout} : (do_flush ? 3'd1 : 3'd0)) -
                    (((!wq_valid || wq_ready) && fq_n != 0) ? 3'd1 : 3'd0);
        end
    end
endmodule
