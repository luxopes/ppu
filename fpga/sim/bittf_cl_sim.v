// Simulacni obal blizky cl_bittf: bittf_fpga_top + axi_own_mux, host (PCIS) a pamet (DDR) jako plne AXI4.
module bittf_cl_sim (
    input  wire          clk,
    input  wire          rst_n,
    // OCL
    input  wire [31:0]   s_axi_awaddr,
    input  wire          s_axi_awvalid,
    output wire          s_axi_awready,
    input  wire [31:0]   s_axi_wdata,
    input  wire [3:0]    s_axi_wstrb,
    input  wire          s_axi_wvalid,
    output wire          s_axi_wready,
    output wire [1:0]    s_axi_bresp,
    output wire          s_axi_bvalid,
    input  wire          s_axi_bready,
    input  wire [31:0]   s_axi_araddr,
    input  wire          s_axi_arvalid,
    output wire          s_axi_arready,
    output wire [31:0]   s_axi_rdata,
    output wire [1:0]    s_axi_rresp,
    output wire          s_axi_rvalid,
    input  wire          s_axi_rready,
    // host (PCIS)
    input  wire [15:0]   h_awid,
    input  wire [63:0]   h_awaddr,
    input  wire [7:0]    h_awlen,
    input  wire          h_awvalid,
    output wire          h_awready,
    input  wire [511:0]  h_wdata,
    input  wire [63:0]   h_wstrb,
    input  wire          h_wlast,
    input  wire          h_wvalid,
    output wire          h_wready,
    output wire [15:0]   h_bid,
    output wire          h_bvalid,
    input  wire          h_bready,
    input  wire [15:0]   h_arid,
    input  wire [63:0]   h_araddr,
    input  wire [7:0]    h_arlen,
    input  wire          h_arvalid,
    output wire          h_arready,
    output wire [15:0]   h_rid,
    output wire [511:0]  h_rdata,
    output wire          h_rlast,
    output wire          h_rvalid,
    input  wire          h_rready,
    // pamet
    output wire [15:0]   m_awid,
    output wire [63:0]   m_awaddr,
    output wire [7:0]    m_awlen,
    output wire          m_awvalid,
    input  wire          m_awready,
    output wire [511:0]  m_wdata,
    output wire [63:0]   m_wstrb,
    output wire          m_wlast,
    output wire          m_wvalid,
    input  wire          m_wready,
    input  wire [15:0]   m_bid,
    input  wire          m_bvalid,
    output wire          m_bready,
    output wire [15:0]   m_arid,
    output wire [63:0]   m_araddr,
    output wire [7:0]    m_arlen,
    output wire          m_arvalid,
    input  wire          m_arready,
    input  wire [15:0]   m_rid,
    input  wire [511:0]  m_rdata,
    input  wire          m_rlast,
    input  wire          m_rvalid,
    output wire          m_rready,
    output wire          own_core
);
    wire [63:0]  c_araddr, c_awaddr;
    wire [7:0]   c_arlen, c_awlen;
    wire [2:0]   c_arsize, c_awsize;
    wire [1:0]   c_arburst, c_awburst;
    wire         c_arvalid, c_arready, c_rvalid, c_rready;
    wire         c_awvalid, c_awready, c_wvalid, c_wready, c_wlast, c_bvalid, c_bready;
    wire [511:0] c_rdata, c_wdata;
    wire [63:0]  c_wstrb;
    wire         busy;
    wire [1:0]   h_bresp, h_rresp;
    wire [2:0]   m_awsize, m_arsize;
    wire [1:0]   m_awburst, m_arburst;

    bittf_fpga_top #(.LB(64), .TM(32), .TN(32)) top (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready), .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .m_axi_araddr(c_araddr), .m_axi_arlen(c_arlen), .m_axi_arsize(c_arsize), .m_axi_arburst(c_arburst),
        .m_axi_arvalid(c_arvalid), .m_axi_arready(c_arready), .m_axi_rdata(c_rdata), .m_axi_rvalid(c_rvalid),
        .m_axi_rready(c_rready), .m_axi_awaddr(c_awaddr), .m_axi_awlen(c_awlen), .m_axi_awsize(c_awsize),
        .m_axi_awburst(c_awburst), .m_axi_awvalid(c_awvalid), .m_axi_awready(c_awready),
        .m_axi_wdata(c_wdata), .m_axi_wstrb(c_wstrb), .m_axi_wlast(c_wlast), .m_axi_wvalid(c_wvalid),
        .m_axi_wready(c_wready), .m_axi_bvalid(c_bvalid), .m_axi_bready(c_bready),
        .busy(busy), .ext_status({30'd0, own_core, 1'b1}));

    axi_own_mux #(.DW(512), .IW(16)) mux (
        .clk(clk), .rst_n(rst_n), .core_req(busy), .own_core(own_core),
        .h_awid(h_awid), .h_awaddr(h_awaddr), .h_awlen(h_awlen), .h_awsize(3'd6), .h_awburst(2'd1),
        .h_awvalid(h_awvalid), .h_awready(h_awready), .h_wdata(h_wdata), .h_wstrb(h_wstrb),
        .h_wlast(h_wlast), .h_wvalid(h_wvalid), .h_wready(h_wready), .h_bid(h_bid), .h_bresp(h_bresp),
        .h_bvalid(h_bvalid), .h_bready(h_bready), .h_arid(h_arid), .h_araddr(h_araddr), .h_arlen(h_arlen),
        .h_arsize(3'd6), .h_arburst(2'd1), .h_arvalid(h_arvalid), .h_arready(h_arready), .h_rid(h_rid),
        .h_rdata(h_rdata), .h_rresp(h_rresp), .h_rlast(h_rlast), .h_rvalid(h_rvalid), .h_rready(h_rready),
        .c_awaddr(c_awaddr), .c_awlen(c_awlen), .c_awsize(c_awsize), .c_awburst(c_awburst),
        .c_awvalid(c_awvalid), .c_awready(c_awready), .c_wdata(c_wdata), .c_wstrb(c_wstrb),
        .c_wlast(c_wlast), .c_wvalid(c_wvalid), .c_wready(c_wready), .c_bvalid(c_bvalid), .c_bready(c_bready),
        .c_araddr(c_araddr), .c_arlen(c_arlen), .c_arsize(c_arsize), .c_arburst(c_arburst),
        .c_arvalid(c_arvalid), .c_arready(c_arready), .c_rdata(c_rdata), .c_rvalid(c_rvalid), .c_rready(c_rready),
        .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_wdata(m_wdata), .m_wstrb(m_wstrb),
        .m_wlast(m_wlast), .m_wvalid(m_wvalid), .m_wready(m_wready), .m_bid(m_bid), .m_bresp(2'd0),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen),
        .m_arsize(m_arsize), .m_arburst(m_arburst), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rid(m_rid), .m_rdata(m_rdata), .m_rresp(2'd0), .m_rlast(m_rlast), .m_rvalid(m_rvalid),
        .m_rready(m_rready));
endmodule
