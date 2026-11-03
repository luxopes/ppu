`timescale 1ns/1ps
// =============================================================================
// FPGA top pro BitLM: AXI4-Lite (napr. AWS F2 OCL BAR0) -> bitlm_core.
// Stejny modul se pouziva v simulaci (tools/sim/bitlm) i v AWS F2 CL obalu
// (fpga/aws_f2/design/cl_ppu_bitlm.sv).
// =============================================================================
module bitlm_fpga_top #(
    parameter LOG2_N = 3,
    parameter LOG2_E = 5,
    parameter LOG2_H = 9,
    parameter LOG2_L = 6,
    parameter LOG2_CORPUS = 22,
    parameter MULTISCALE = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [31:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,
    output wire        busy
);
    wire        mmio_we, mmio_re, mmio_rvalid;
    wire [23:0] mmio_addr;
    wire [31:0] mmio_wdata, mmio_rdata;

    // synchronni reset (rst_main_n na F2 je synchronni s clk_main_a0)
    reg rst_q;
    always @(posedge clk) rst_q <= ~rst_n;

    axi_lite_mmio #(.AW(24)) bridge (
        .aclk(clk), .aresetn(rst_n),
        .s_awaddr(s_axi_awaddr), .s_awvalid(s_axi_awvalid), .s_awready(s_axi_awready),
        .s_wdata(s_axi_wdata), .s_wstrb(s_axi_wstrb), .s_wvalid(s_axi_wvalid), .s_wready(s_axi_wready),
        .s_bresp(s_axi_bresp), .s_bvalid(s_axi_bvalid), .s_bready(s_axi_bready),
        .s_araddr(s_axi_araddr), .s_arvalid(s_axi_arvalid), .s_arready(s_axi_arready),
        .s_rdata(s_axi_rdata), .s_rresp(s_axi_rresp), .s_rvalid(s_axi_rvalid), .s_rready(s_axi_rready),
        .mmio_we(mmio_we), .mmio_re(mmio_re), .mmio_addr(mmio_addr),
        .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata), .mmio_rvalid(mmio_rvalid)
    );

    bitlm_core #(
        .LOG2_N(LOG2_N), .LOG2_E(LOG2_E), .LOG2_H(LOG2_H), .LOG2_L(LOG2_L),
        .LOG2_CORPUS(LOG2_CORPUS), .MULTISCALE(MULTISCALE)
    ) core (
        .clk(clk), .rst(rst_q),
        .mmio_we(mmio_we), .mmio_re(mmio_re), .mmio_addr(mmio_addr),
        .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata), .mmio_rvalid(mmio_rvalid),
        .busy(busy)
    );
endmodule
