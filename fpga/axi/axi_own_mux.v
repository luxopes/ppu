// Prepinac vlastnictvi pameti: dva AXI4 mastery (host = PCIS, jadro) -> jeden slave (DDR).
// V kazdem okamziku patri pamet jen jednomu z nich, takze neni potreba arbitraz po transakcich.
// Jadro dostane pamet, kdyz o ni zada (core_req = jadro bezi) a host nema nic rozpracovaneho;
// host ji dostane zpet, kdyz jadro skonci a nema nic rozpracovaneho. Behem zadosti jadra se
// hostu nove pozadavky neprijimaji. Jadro pouziva ID 0 a prenosy o 1 beatu.
module axi_own_mux #(
    parameter DW = 512,
    parameter IW = 16
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              core_req,
    output reg               own_core,
    // host (plny AXI4)
    input  wire [IW-1:0]     h_awid,
    input  wire [63:0]       h_awaddr,
    input  wire [7:0]        h_awlen,
    input  wire [2:0]        h_awsize,
    input  wire [1:0]        h_awburst,
    input  wire              h_awvalid,
    output wire              h_awready,
    input  wire [DW-1:0]     h_wdata,
    input  wire [DW/8-1:0]   h_wstrb,
    input  wire              h_wlast,
    input  wire              h_wvalid,
    output wire              h_wready,
    output wire [IW-1:0]     h_bid,
    output wire [1:0]        h_bresp,
    output wire              h_bvalid,
    input  wire              h_bready,
    input  wire [IW-1:0]     h_arid,
    input  wire [63:0]       h_araddr,
    input  wire [7:0]        h_arlen,
    input  wire [2:0]        h_arsize,
    input  wire [1:0]        h_arburst,
    input  wire              h_arvalid,
    output wire              h_arready,
    output wire [IW-1:0]     h_rid,
    output wire [DW-1:0]     h_rdata,
    output wire [1:0]        h_rresp,
    output wire              h_rlast,
    output wire              h_rvalid,
    input  wire              h_rready,
    // jadro
    input  wire [63:0]       c_awaddr,
    input  wire [7:0]        c_awlen,
    input  wire [2:0]        c_awsize,
    input  wire [1:0]        c_awburst,
    input  wire              c_awvalid,
    output wire              c_awready,
    input  wire [DW-1:0]     c_wdata,
    input  wire [DW/8-1:0]   c_wstrb,
    input  wire              c_wlast,
    input  wire              c_wvalid,
    output wire              c_wready,
    output wire              c_bvalid,
    input  wire              c_bready,
    input  wire [63:0]       c_araddr,
    input  wire [7:0]        c_arlen,
    input  wire [2:0]        c_arsize,
    input  wire [1:0]        c_arburst,
    input  wire              c_arvalid,
    output wire              c_arready,
    output wire [DW-1:0]     c_rdata,
    output wire              c_rvalid,
    input  wire              c_rready,
    // pamet
    output wire [IW-1:0]     m_awid,
    output wire [63:0]       m_awaddr,
    output wire [7:0]        m_awlen,
    output wire [2:0]        m_awsize,
    output wire [1:0]        m_awburst,
    output wire              m_awvalid,
    input  wire              m_awready,
    output wire [DW-1:0]     m_wdata,
    output wire [DW/8-1:0]   m_wstrb,
    output wire              m_wlast,
    output wire              m_wvalid,
    input  wire              m_wready,
    input  wire [IW-1:0]     m_bid,
    input  wire [1:0]        m_bresp,
    input  wire              m_bvalid,
    output wire              m_bready,
    output wire [IW-1:0]     m_arid,
    output wire [63:0]       m_araddr,
    output wire [7:0]        m_arlen,
    output wire [2:0]        m_arsize,
    output wire [1:0]        m_arburst,
    output wire              m_arvalid,
    input  wire              m_arready,
    input  wire [IW-1:0]     m_rid,
    input  wire [DW-1:0]     m_rdata,
    input  wire [1:0]        m_rresp,
    input  wire              m_rlast,
    input  wire              m_rvalid,
    output wire              m_rready
);
    // host smi zadavat nove adresy jen kdyz vlastni pamet a jadro o ni nezada
    wire h_ok = !own_core && !core_req;
    wire h_on = !own_core;

    assign m_awid    = own_core ? {IW{1'b0}} : h_awid;
    assign m_awaddr  = own_core ? c_awaddr  : h_awaddr;
    assign m_awlen   = own_core ? c_awlen   : h_awlen;
    assign m_awsize  = own_core ? c_awsize  : h_awsize;
    assign m_awburst = own_core ? c_awburst : h_awburst;
    assign m_awvalid = own_core ? c_awvalid : (h_awvalid && h_ok);
    assign h_awready = h_ok && m_awready;
    assign c_awready = own_core && m_awready;

    assign m_wdata   = own_core ? c_wdata  : h_wdata;
    assign m_wstrb   = own_core ? c_wstrb  : h_wstrb;
    assign m_wlast   = own_core ? c_wlast  : h_wlast;
    assign m_wvalid  = own_core ? c_wvalid : (h_wvalid && h_on);
    assign h_wready  = h_on && m_wready;
    assign c_wready  = own_core && m_wready;

    assign h_bid     = m_bid;
    assign h_bresp   = m_bresp;
    assign h_bvalid  = h_on && m_bvalid;
    assign c_bvalid  = own_core && m_bvalid;
    assign m_bready  = own_core ? c_bready : h_bready;

    assign m_arid    = own_core ? {IW{1'b0}} : h_arid;
    assign m_araddr  = own_core ? c_araddr  : h_araddr;
    assign m_arlen   = own_core ? c_arlen   : h_arlen;
    assign m_arsize  = own_core ? c_arsize  : h_arsize;
    assign m_arburst = own_core ? c_arburst : h_arburst;
    assign m_arvalid = own_core ? c_arvalid : (h_arvalid && h_ok);
    assign h_arready = h_ok && m_arready;
    assign c_arready = own_core && m_arready;

    assign h_rid     = m_rid;
    assign h_rdata   = m_rdata;
    assign h_rresp   = m_rresp;
    assign h_rlast   = m_rlast;
    assign h_rvalid  = h_on && m_rvalid;
    assign c_rdata   = m_rdata;
    assign c_rvalid  = own_core && m_rvalid;
    assign m_rready  = own_core ? c_rready : h_rready;

    // rozpracovane transakce aktualniho vlastnika (zapis: AW..B, cteni: AR..posledni R)
    reg [15:0] n_wr, n_rd;
    wire aw_hs = m_awvalid && m_awready;
    wire b_hs  = m_bvalid && m_bready;
    wire ar_hs = m_arvalid && m_arready;
    wire r_end = m_rvalid && m_rready && (own_core || m_rlast);
    wire idle  = n_wr == 0 && n_rd == 0 && !aw_hs && !ar_hs;

    always @(posedge clk) begin
        if (!rst_n) begin
            own_core <= 1'b0;
            n_wr <= 0;
            n_rd <= 0;
        end else begin
            n_wr <= n_wr + (aw_hs ? 16'd1 : 16'd0) - (b_hs ? 16'd1 : 16'd0);
            n_rd <= n_rd + (ar_hs ? 16'd1 : 16'd0) - (r_end ? 16'd1 : 16'd0);
            if (!own_core && core_req && idle) own_core <= 1'b1;
            else if (own_core && !core_req && idle) own_core <= 1'b0;
        end
    end
endmodule
