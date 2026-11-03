`timescale 1ns/1ps
// FPGA top: expose AXI4-Lite slave ports for PCIe/IP to connect to,
// bridge AXI-Lite to PPU MMIO and instantiate PPU core.
module fpga_top(
    input  wire        clk,
    input  wire        rst_n,
    // AXI4-Lite slave interface (host -> FPGA)
    input  wire [31:0] S_AXI_AWADDR,
    input  wire        S_AXI_AWVALID,
    output wire        S_AXI_AWREADY,
    input  wire [31:0] S_AXI_WDATA,
    input  wire [3:0]  S_AXI_WSTRB,
    input  wire        S_AXI_WVALID,
    output wire        S_AXI_WREADY,
    output wire [1:0]  S_AXI_BRESP,
    output wire        S_AXI_BVALID,
    input  wire        S_AXI_BREADY,
    input  wire [31:0] S_AXI_ARADDR,
    input  wire        S_AXI_ARVALID,
    output wire        S_AXI_ARREADY,
    output wire [31:0] S_AXI_RDATA,
    output wire [1:0]  S_AXI_RRESP,
    output wire        S_AXI_RVALID,
    input  wire        S_AXI_RREADY,

    // debug UART pins (optional)
    output wire        uart_tx
);

    // instantiate axi_lite_slave
    wire mmio_we;
    wire [7:0] mmio_addr;
    wire [31:0] mmio_wdata;
    wire [31:0] mmio_rdata;

    axi_lite_slave axi_slave_inst(
        .ACLK(clk),
        .ARESETn(rst_n),
        .AWADDR(S_AXI_AWADDR),
        .AWVALID(S_AXI_AWVALID),
        .AWREADY(S_AXI_AWREADY),
        .WDATA(S_AXI_WDATA),
        .WSTRB(S_AXI_WSTRB),
        .WVALID(S_AXI_WVALID),
        .WREADY(S_AXI_WREADY),
        .BRESP(S_AXI_BRESP),
        .BVALID(S_AXI_BVALID),
        .BREADY(S_AXI_BREADY),
        .ARADDR(S_AXI_ARADDR),
        .ARVALID(S_AXI_ARVALID),
        .ARREADY(S_AXI_ARREADY),
        .RDATA(S_AXI_RDATA),
        .RRESP(S_AXI_RRESP),
        .RVALID(S_AXI_RVALID),
        .RREADY(S_AXI_RREADY),
        .mmio_we(mmio_we),
        .mmio_addr(mmio_addr),
        .mmio_wdata(mmio_wdata),
        .mmio_rdata(mmio_rdata)
    );

    // instantiate ppu_top
    wire [31:0] accel_result;

    ppu_top dut(
        .clk(clk),
        .rst(~rst_n), // ppu_top expects active-high rst
        .mmio_we(mmio_we),
        .mmio_addr(mmio_addr),
        .mmio_wdata(mmio_wdata),
        .mmio_rdata(mmio_rdata),
        .accel_result_out(accel_result)
    );

    // simple uart tx tied low for now
    assign uart_tx = 1'b1;

endmodule
