`timescale 1ns/1ps
// Top-level PPU module: instantiate core, imem, dmem and AI accel
//
// Opravy 2026-11-03:
//  - jadro je jednocyklove -> CPU cteni dmem je asynchronni (drive registrovane,
//    LW tak vracel data o takt pozdeji / z predchozi adresy),
//  - zapisy a cteni CPU do MMIO (>= 0x1000_0000) jsou smerovany na akcelerator
//    (drive "not implemented"); pri soucasnem zapisu z AXI ma prednost AXI,
//  - inicializace pameti z .mem souboru (parametry IMEM_INIT/DMEM_INIT),
//  - deklarace integer v pojmenovanem bloku (drive neplatny Verilog-2001).
module ppu_top #(
    parameter IMEM_INIT = "",
    parameter DMEM_INIT = ""
) (
    input wire clk,
    input wire rst,
    // simple MMIO interface (from AXI-Lite bridge)
    input  wire        mmio_we,
    input  wire [7:0]  mmio_addr,
    input  wire [31:0] mmio_wdata,
    output wire [31:0] mmio_rdata,
    output wire [31:0] accel_result_out
);

    localparam MMIO_BASE = 32'h1000_0000;

    // simple instruction memory (words)
    reg [31:0] imem [0:255];
    // simple data memory (words)
    reg [31:0] dmem [0:1023];

    wire [31:0] instr;
    wire [31:0] pc;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [3:0]  mem_wstrb;
    wire        mem_write;
    wire        mem_read;
    wire [31:0] mem_rdata;

    wire        accel_start_wire;
    wire [31:0] accel_base1_wire;
    wire [31:0] accel_base2_wire;
    wire [31:0] accel_len_wire;

    core cpu(
        .clk(clk),
        .rst(rst),
        .instr(instr),
        .pc(pc),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_write(mem_write),
        .mem_read(mem_read),
        .mem_rdata(mem_rdata),
        .accel_start(accel_start_wire),
        .accel_base1(accel_base1_wire),
        .accel_base2(accel_base2_wire),
        .accel_len(accel_len_wire)
    );

    assign instr = imem[pc[9:2]];

    wire is_mmio = (mem_addr >= MMIO_BASE);

    // MMIO akceleratoru: AXI ma prednost, jinak zapis z CPU
    wire        cpu_mmio_we = mem_write && is_mmio;
    wire        acc_mmio_we = mmio_we | cpu_mmio_we;
    // adresa: pri zapisu z CPU adresa CPU, jinak adresa z AXI (zapis i cteni)
    wire [7:0]  acc_mmio_addr_rd = (cpu_mmio_we && !mmio_we) ? mem_addr[7:0] : mmio_addr;
    wire [31:0] acc_mmio_wdata = mmio_we ? mmio_wdata : mem_wdata;

    wire        accel_mem_rd_en;
    wire [31:0] accel_mem_rd_addr;
    reg  [31:0] accel_mem_rd_data_reg;
    wire [31:0] accel_mmio_rdata;
    wire [31:0] accel_cpu_rdata;
    wire [31:0] accel_result;

    ai_accel accel(
        .clk(clk),
        .rst(rst),
        .mmio_we(acc_mmio_we),
        .mmio_addr(acc_mmio_addr_rd),
        .mmio_wdata(acc_mmio_wdata),
        .mmio_rdata(accel_mmio_rdata),
        .cpu_raddr(mem_addr[7:0]),
        .cpu_rdata(accel_cpu_rdata),
        .mem_rd_en(accel_mem_rd_en),
        .mem_rd_addr(accel_mem_rd_addr),
        .mem_rd_data(accel_mem_rd_data_reg),
        .start_in(accel_start_wire),
        .base1_in(accel_base1_wire),
        .base2_in(accel_base2_wire),
        .len_in(accel_len_wire),
        .result_out(accel_result)
    );

    assign accel_result_out = accel_result;
    assign mmio_rdata = accel_mmio_rdata;

    // CPU cteni: asynchronni (jednocyklove jadro)
    assign mem_rdata = is_mmio ? accel_cpu_rdata : dmem[mem_addr[11:2]];

    always @(posedge clk) begin
        // akcelerator: registrovane cteni (data 2 takty po nastaveni adresy)
        if (accel_mem_rd_en)
            accel_mem_rd_data_reg <= dmem[accel_mem_rd_addr[11:2]];
        // CPU zapis do dmem
        if (mem_write && !is_mmio)
            dmem[mem_addr[11:2]] <= mem_wdata;
    end

    initial begin : mem_init
        integer i;
        for (i = 0; i < 256; i = i + 1) imem[i] = 32'b0;
        for (i = 0; i < 1024; i = i + 1) dmem[i] = 32'b0;
        accel_mem_rd_data_reg = 32'd0;
        if (IMEM_INIT != "") $readmemh(IMEM_INIT, imem);
        if (DMEM_INIT != "") $readmemh(DMEM_INIT, dmem);
    end

endmodule
