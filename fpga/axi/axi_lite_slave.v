`timescale 1ns/1ps
// AXI4-Lite slave pro puvodni PPU (8-bit MMIO adresa, registrovane cteni).
//
// Oprava 2026-11-03: puvodni verze pri cteni vubec nepredavala ARADDR do
// mmio_addr (vracela registr podle posledniho zapisu) a data brala bez cekani
// na registrovany vystup akceleratoru. Nyni obal nad axi_lite_mmio: cteni
// nastavi adresu a data se vraci po 2 taktech (ai_accel ma 1 takt latenci).
module axi_lite_slave(
    input  wire        ACLK,
    input  wire        ARESETn,
    input  wire [31:0] AWADDR,
    input  wire        AWVALID,
    output wire        AWREADY,
    input  wire [31:0] WDATA,
    input  wire [3:0]  WSTRB,
    input  wire        WVALID,
    output wire        WREADY,
    output wire [1:0]  BRESP,
    output wire        BVALID,
    input  wire        BREADY,
    input  wire [31:0] ARADDR,
    input  wire        ARVALID,
    output wire        ARREADY,
    output wire [31:0] RDATA,
    output wire [1:0]  RRESP,
    output wire        RVALID,
    input  wire        RREADY,

    output wire        mmio_we,
    output wire [7:0]  mmio_addr,
    output wire [31:0] mmio_wdata,
    input  wire [31:0] mmio_rdata
);
    wire mmio_re;
    reg  [1:0] rd_pipe;

    // cteni: adresa plati od taktu po mmio_re, registrovana data o takt pozdeji
    always @(posedge ACLK) begin
        if (!ARESETn) rd_pipe <= 2'b00;
        else rd_pipe <= {rd_pipe[0], mmio_re};
    end

    axi_lite_mmio #(.AW(8)) bridge (
        .aclk(ACLK), .aresetn(ARESETn),
        .s_awaddr(AWADDR), .s_awvalid(AWVALID), .s_awready(AWREADY),
        .s_wdata(WDATA), .s_wstrb(WSTRB), .s_wvalid(WVALID), .s_wready(WREADY),
        .s_bresp(BRESP), .s_bvalid(BVALID), .s_bready(BREADY),
        .s_araddr(ARADDR), .s_arvalid(ARVALID), .s_arready(ARREADY),
        .s_rdata(RDATA), .s_rresp(RRESP), .s_rvalid(RVALID), .s_rready(RREADY),
        .mmio_we(mmio_we), .mmio_re(mmio_re), .mmio_addr(mmio_addr),
        .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata), .mmio_rvalid(rd_pipe[1])
    );
endmodule
