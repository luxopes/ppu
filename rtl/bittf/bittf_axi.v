// BitTF: most z NP radkovych portu jadra na jeden AXI4 master (datova sirka = radek, LB bajtu).
// Kazdy pozadavek = jeden AXI prenos o 1 beatu (ARLEN/AWLEN = 0). Jedno ID -> odpovedi v poradi;
// fronty s cislem portu rozvadi R a B zpet na porty. Zapisy prichazeji jen z portu 0.
// Adresa AXI = mem_base + radek * LB.
module bittf_axi #(
    parameter NP = 5,
    parameter LB = 64,
    parameter OUTST = 64         // max. cteni v letu
) (
    input  wire                clk,
    input  wire                rst,
    input  wire [63:0]         mem_base,
    // porty jadra
    input  wire [NP-1:0]       p_rq_valid,
    output reg  [NP-1:0]       p_rq_ready,
    input  wire [NP-1:0]       p_rq_we,
    input  wire [NP*26-1:0]    p_rq_line,
    input  wire [NP*LB*8-1:0]  p_rq_wdata,
    input  wire [NP*LB-1:0]    p_rq_wstrb,
    output reg  [NP-1:0]       p_rs_valid,
    output reg  [NP*LB*8-1:0]  p_rs_data,
    output reg  [NP-1:0]       p_wack,
    // AXI4 master
    output reg  [63:0]         m_araddr,
    output wire [7:0]          m_arlen,
    output wire [2:0]          m_arsize,
    output wire [1:0]          m_arburst,
    output reg                 m_arvalid,
    input  wire                m_arready,
    input  wire [LB*8-1:0]     m_rdata,
    input  wire                m_rvalid,
    output wire                m_rready,
    output reg  [63:0]         m_awaddr,
    output wire [7:0]          m_awlen,
    output wire [2:0]          m_awsize,
    output wire [1:0]          m_awburst,
    output reg                 m_awvalid,
    input  wire                m_awready,
    output reg  [LB*8-1:0]     m_wdata,
    output reg  [LB-1:0]       m_wstrb,
    output wire                m_wlast,
    output reg                 m_wvalid,
    input  wire                m_wready,
    input  wire                m_bvalid,
    output wire                m_bready
);
    localparam LBB = $clog2(LB);
    localparam PW = $clog2(NP);
    assign m_arlen = 8'd0;
    assign m_awlen = 8'd0;
    assign m_arsize = LBB[2:0];
    assign m_awsize = LBB[2:0];
    assign m_arburst = 2'b01;
    assign m_awburst = 2'b01;
    assign m_wlast = 1'b1;
    assign m_rready = 1'b1;
    assign m_bready = 1'b1;

    // fronta portu ctenych radku
    reg [PW-1:0] rf [0:OUTST-1];
    reg [$clog2(OUTST):0] rf_n;
    reg [$clog2(OUTST)-1:0] rf_w, rf_r;
    // fronta zapisu (jen pro pocty, vsechny z portu 0)
    reg [15:0] wf_n;

    // vyber cteni: round-robin
    reg [PW-1:0] rr;
    reg [PW-1:0] sel;
    reg sel_ok;
    integer k, j;
    always @(*) begin
        sel = 0;
        sel_ok = 1'b0;
        for (k = NP - 1; k >= 0; k = k - 1) begin
            j = (rr + 1 + k) % NP;
            if (p_rq_valid[j] && !p_rq_we[j]) begin
                sel = j[PW-1:0];
                sel_ok = 1'b1;
            end
        end
    end
    wire ar_free = !m_arvalid || m_arready;
    wire rd_go = sel_ok && ar_free && rf_n < OUTST;
    wire aw_free = (!m_awvalid || m_awready) && (!m_wvalid || m_wready);
    wire wr_go = p_rq_valid[0] && p_rq_we[0] && aw_free;

    always @(*) begin
        p_rq_ready = 0;
        if (rd_go) p_rq_ready[sel] = 1'b1;
        if (wr_go) p_rq_ready[0] = 1'b1;
    end

    always @(posedge clk) begin
        if (rst) begin
            m_arvalid <= 1'b0;
            m_awvalid <= 1'b0;
            m_wvalid <= 1'b0;
            rf_n <= 0;
            rf_w <= 0;
            rf_r <= 0;
            rr <= 0;
            p_rs_valid <= 0;
            p_wack <= 0;
        end else begin
            if (m_arvalid && m_arready) m_arvalid <= 1'b0;
            if (m_awvalid && m_awready) m_awvalid <= 1'b0;
            if (m_wvalid && m_wready) m_wvalid <= 1'b0;
            if (rd_go) begin
                m_arvalid <= 1'b1;
                m_araddr <= mem_base + {p_rq_line[26 * sel +: 26], {LBB{1'b0}}};
                rf[rf_w] <= sel;
                rf_w <= rf_w + 1'b1;
                rr <= sel;
            end
            if (wr_go) begin
                m_awvalid <= 1'b1;
                m_awaddr <= mem_base + {p_rq_line[25:0], {LBB{1'b0}}};
                m_wvalid <= 1'b1;
                m_wdata <= p_rq_wdata[LB*8-1:0];
                m_wstrb <= p_rq_wstrb[LB-1:0];
            end
            // odpovedi
            p_rs_valid <= 0;
            if (m_rvalid) begin
                p_rs_valid[rf[rf_r]] <= 1'b1;
                p_rs_data[LB*8*rf[rf_r] +: LB*8] <= m_rdata;
                rf_r <= rf_r + 1'b1;
            end
            rf_n <= rf_n + (rd_go ? 1'b1 : 1'b0) - (m_rvalid ? 1'b1 : 1'b0);
            p_wack <= {{(NP-1){1'b0}}, m_bvalid};
        end
    end
endmodule
