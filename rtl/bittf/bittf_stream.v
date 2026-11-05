// BitTF v2: proudova cteni operandu. Prochazi 4-D smycku (jako ISA) po "taktech" (beat): ve vektorovem
// rezimu az VL po sobe jdoucich prvku urovne 0 (krok s0 = 1, nebo s0 = 0 = broadcast), jinak 1 prvek.
// Cte cele radky pameti (LB bajtu) pres vlastni port, opakovane pouzity radek necte znovu.
// Gather: offset prvku += idx * gscale, idx prichazi z dalsiho proudu (jen skalarni rezim).
module bittf_stream #(
    parameter VL = 8,           // max. prvku na takt
    parameter LB = 64,          // bajtu na radek
    parameter FD = 32           // hloubka front (radky v letu)
) (
    input  wire              clk,
    input  wire              rst,
    // konfigurace (stabilni po dobu instrukce), start = 1 takt
    input  wire              start,
    input  wire [31:0]       base,
    input  wire [1:0]        lw,
    input  wire              uns,
    input  wire              vec,        // vektorovy rezim (jinak 1 prvek na takt)
    input  wire [127:0]      cnt,        // 4 x 32 bitu, uroven 0 dole
    input  wire [255:0]      str,        // 4 x 64 bitu kroky v prvcich
    input  wire              gath,
    input  wire signed [63:0] gscale,
    input  wire              gidx_valid,
    output wire              gidx_ready,
    input  wire signed [63:0] gidx,
    // radkovy port (jen cteni)
    output reg               rq_valid,
    input  wire              rq_ready,
    output reg  [25:0]       rq_line,
    input  wire              rs_valid,
    input  wire [LB*8-1:0]   rs_data,
    // vystup
    output wire              o_valid,
    input  wire              o_ready,
    output reg  [VL*64-1:0]  o_data,
    output wire [6:0]        o_n,
    output wire              busy
);
    localparam LBB = $clog2(LB);

    // ------------------------------------------------------------ generator taktu
    reg [31:0] c [0:3];
    reg [31:0] i [0:3];
    reg signed [63:0] s [0:3];
    reg signed [63:0] off [0:3];
    reg running, have_last;
    reg [25:0] last_line;
    integer k;

    wire signed [63:0] s0 = s[0];
    // delka taktu na urovni 0
    wire [31:0] rem0 = c[0] - i[0];
    // takt nesmi presahnout radek: nejvys LB >> lw prvku
    wire [31:0] vmax = ((LB >> lw) < VL) ? (LB >> lw) : VL;
    wire [6:0]  blen = (!vec) ? 7'd1 : (rem0 >= vmax) ? vmax[6:0] : rem0[6:0];
    wire signed [63:0] eoff = off[0] + (gath ? gidx * gscale : 64'sd0);
    wire [31:0] baddr = base + (eoff[31:0] << lw);
    wire [31:0] bytes = (s0 == 0) ? (32'd1 << lw) : ({25'd0, blen} << lw);
    wire [31:0] eaddr_last = baddr + bytes - 32'd1;
    wire [25:0] L0 = baddr[31:LBB], L1 = eaddr_last[31:LBB];
    wire new0 = !have_last || L0 != last_line;
    wire new1 = L1 != L0;
    wire [1:0] nreq = {1'b0, new0} + {1'b0, new1};

    // fronta popisu taktu a kredit radku
    localparam DW = LBB + 7 + 3;
    reg  [DW-1:0] dq [0:FD-1];
    reg  [$clog2(FD):0] dq_n;
    reg  [$clog2(FD)-1:0] dq_w, dq_r;
    reg  [$clog2(FD)+1:0] credit;     // radku, ktere jeste smime pozadat (misto ve fronte odpovedi)
    reg  pend1;                       // ceka druhy pozadavek (L1) taktu
    reg  [25:0] pend_line;

    wire last_beat = (i[0] + blen >= c[0]) && (i[1] + 1 >= c[1]) && (i[2] + 1 >= c[2]) && (i[3] + 1 >= c[3]);
    wire can_gen = running && !pend1 && dq_n < FD - 1 && credit >= nreq && (!gath || gidx_valid) &&
                   (!rq_valid || rq_ready);
    assign gidx_ready = can_gen && gath;
    assign busy = running || dq_n != 0 || pend1;

    // posun smycek o takt
    reg [2:0] adv;
    always @(*) begin
        if (i[0] + blen < c[0]) adv = 3'd0;
        else if (i[1] + 1 < c[1]) adv = 3'd1;
        else if (i[2] + 1 < c[2]) adv = 3'd2;
        else if (i[3] + 1 < c[3]) adv = 3'd3;
        else adv = 3'd4;
    end

    wire pop_desc;
    wire [DW-1:0] dq_head = dq[dq_r];

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0;
            rq_valid <= 1'b0;
            dq_n <= 0;
            dq_w <= 0;
            dq_r <= 0;
            pend1 <= 1'b0;
        end else begin
            if (rq_valid && rq_ready) rq_valid <= 1'b0;
            if (start) begin
                for (k = 0; k < 4; k = k + 1) begin
                    c[k] <= cnt[32 * k +: 32];
                    s[k] <= str[64 * k +: 64];
                    i[k] <= 32'd0;
                    off[k] <= 64'sd0;
                end
                running <= 1'b1;
                have_last <= 1'b0;
            end else if (pend1 && (!rq_valid || rq_ready)) begin
                rq_valid <= 1'b1;
                rq_line <= pend_line;
                pend1 <= 1'b0;
            end else if (can_gen) begin
                // pozadavky radku
                if (new0) begin
                    rq_valid <= 1'b1;
                    rq_line <= L0;
                    if (new1) begin
                        pend1 <= 1'b1;
                        pend_line <= L1;
                    end
                end else if (new1) begin
                    rq_valid <= 1'b1;
                    rq_line <= L1;
                end
                if (new0 || new1) begin
                    have_last <= 1'b1;
                    last_line <= L1;
                end
                // popis taktu
                dq[dq_w] <= {baddr[LBB-1:0], blen, (s0 == 0), new0, new1};
                dq_w <= dq_w + 1'b1;
                // smycky
                for (k = 0; k < 4; k = k + 1) begin
                    if (k < adv) i[k] <= 32'd0;
                    else if (k == adv) i[k] <= i[k] + ((k == 0) ? {25'd0, blen} : 32'd1);
                end
                for (k = 0; k < 4; k = k + 1)
                    if (k <= adv && adv < 4)
                        off[k] <= off[adv] + ((adv == 0) ? (vec ? ((s0 == 0) ? 64'sd0 : {57'd0, blen}) : s[0]) : s[adv]);
                if (last_beat) running <= 1'b0;
            end
            // pocty ve fronte popisu a kredity
            dq_n <= dq_n + ((can_gen && !start && !(pend1 && (!rq_valid || rq_ready))) ? 1'b1 : 1'b0) - (pop_desc ? 1'b1 : 1'b0);
            if (pop_desc) dq_r <= dq_r + 1'b1;
        end
    end

    // ------------------------------------------------------------ fronta odpovedi (radky)
    reg  [LB*8-1:0] lq [0:FD-1];
    reg  [$clog2(FD):0] lq_n;
    reg  [$clog2(FD)-1:0] lq_w, lq_r;
    wire pop_line;
    wire [1:0] npop;

    // ------------------------------------------------------------ extrakce
    wire [LBB-1:0] h_off = dq_head[DW-1 -: LBB];
    wire [6:0]     h_len = dq_head[9:3];
    wire           h_bc = dq_head[2];
    wire           h_new0 = dq_head[1];
    wire           h_new1 = dq_head[0];
    reg  [LB*8-1:0] cur;
    wire [1:0] need = {1'b0, h_new0} + {1'b0, h_new1};
    wire lines_ok = lq_n >= need;
    assign o_valid = dq_n != 0 && lines_ok;
    assign pop_desc = o_valid && o_ready;
    assign npop = pop_desc ? need : 2'd0;
    assign o_n = h_len;

    // okno 2 radku: dolni = radek s prvnim prvkem
    wire [LB*8-1:0] w0 = h_new0 ? lq[lq_r] : cur;
    wire [LB*8-1:0] w1 = h_new0 ? (h_new1 ? lq[lq_r + 1'b1] : {LB*8{1'b0}}) : (h_new1 ? lq[lq_r] : {LB*8{1'b0}});
    localparam WW = (2 * LB * 8 > VL * 64) ? 2 * LB * 8 : VL * 64;
    wire [WW-1:0] win = {{(WW - 2*LB*8){1'b0}}, w1, w0};
    wire [WW-1:0] wsh = win >> ({h_off, 3'b000});
    integer l;
    reg [63:0] e;
    always @(*) begin
        for (l = 0; l < VL; l = l + 1) begin
            case (lw)
                2'd0: e = h_bc ? wsh[7:0] : wsh[8 * l +: 8];
                2'd1: e = h_bc ? wsh[15:0] : wsh[16 * l +: 16];
                2'd2: e = h_bc ? wsh[31:0] : wsh[32 * l +: 32];
                default: e = h_bc ? wsh[63:0] : wsh[64 * l +: 64];
            endcase
            case (lw)
                2'd0: o_data[64 * l +: 64] = uns ? {56'd0, e[7:0]} : {{56{e[7]}}, e[7:0]};
                2'd1: o_data[64 * l +: 64] = uns ? {48'd0, e[15:0]} : {{48{e[15]}}, e[15:0]};
                2'd2: o_data[64 * l +: 64] = uns ? {32'd0, e[31:0]} : {{32{e[31]}}, e[31:0]};
                default: o_data[64 * l +: 64] = e;
            endcase
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            lq_n <= 0;
            lq_w <= 0;
            lq_r <= 0;
            credit <= FD;
        end else begin
            if (rs_valid) begin
                lq[lq_w] <= rs_data;
                lq_w <= lq_w + 1'b1;
            end
            lq_n <= lq_n + (rs_valid ? 1'b1 : 1'b0) - npop;
            lq_r <= lq_r + npop;
            if (pop_desc && (h_new0 || h_new1))
                cur <= h_new1 ? (h_new0 ? lq[lq_r + 1'b1] : lq[lq_r]) : lq[lq_r];
            // kredit: vygenerovany takt rezervuje sve radky, vyzvednuti je vraci
            credit <= credit - ((can_gen && !start) ? nreq : 2'd0) + npop;
        end
    end
endmodule
