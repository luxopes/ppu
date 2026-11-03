`timescale 1ns/1ps
// Wrapper pro Verilator: ppu_top s programem z fpga/mem a MMIO porty ven
module tb_ppu(
    input  wire        clk,
    input  wire        rst,
    input  wire        mmio_we,
    input  wire [7:0]  mmio_addr,
    input  wire [31:0] mmio_wdata,
    output wire [31:0] mmio_rdata,
    output wire [31:0] accel_result_out
);

    ppu_top #(
        .IMEM_INIT("../../fpga/mem/imem.mem"),
        .DMEM_INIT("../../fpga/mem/dmem.mem")
    ) dut(
        .clk(clk), .rst(rst),
        .mmio_we(mmio_we), .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata),
        .mmio_rdata(mmio_rdata),
        .accel_result_out(accel_result_out)
    );

endmodule
