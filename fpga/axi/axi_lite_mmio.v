`timescale 1ns/1ps
// =============================================================================
// AXI4-Lite slave -> jednoduche MMIO s promennou latenci cteni.
// Zapis: AW a W mohou prijit v libovolnem poradi; po prijeti obou se vygeneruje
//        jeden puls mmio_we a odpoved B. Cteni: puls mmio_re, ceka se na
//        mmio_rvalid, pak R. Zapis a cteni se obsluhuji postupne (jedna sbernice).
// =============================================================================
module axi_lite_mmio #(
    parameter AW = 24
) (
    input  wire          aclk,
    input  wire          aresetn,
    // AXI4-Lite
    input  wire [31:0]   s_awaddr,
    input  wire          s_awvalid,
    output reg           s_awready,
    input  wire [31:0]   s_wdata,
    input  wire [3:0]    s_wstrb,
    input  wire          s_wvalid,
    output reg           s_wready,
    output wire [1:0]    s_bresp,
    output reg           s_bvalid,
    input  wire          s_bready,
    input  wire [31:0]   s_araddr,
    input  wire          s_arvalid,
    output reg           s_arready,
    output reg  [31:0]   s_rdata,
    output wire [1:0]    s_rresp,
    output reg           s_rvalid,
    input  wire          s_rready,
    // MMIO
    output reg           mmio_we,
    output reg           mmio_re,
    output reg  [AW-1:0] mmio_addr,
    output reg  [31:0]   mmio_wdata,
    input  wire [31:0]   mmio_rdata,
    input  wire          mmio_rvalid
);
    assign s_bresp = 2'b00;
    assign s_rresp = 2'b00;

    reg          aw_ok, w_ok;
    reg [AW-1:0] aw_q;
    reg [31:0]   w_q;
    reg          rd_wait;

    always @(posedge aclk) begin
        if (!aresetn) begin
            s_awready <= 1'b0; s_wready <= 1'b0; s_bvalid <= 1'b0;
            s_arready <= 1'b0; s_rvalid <= 1'b0; s_rdata <= 32'd0;
            aw_ok <= 1'b0; w_ok <= 1'b0; aw_q <= {AW{1'b0}}; w_q <= 32'd0;
            rd_wait <= 1'b0;
            mmio_we <= 1'b0; mmio_re <= 1'b0; mmio_addr <= {AW{1'b0}}; mmio_wdata <= 32'd0;
        end else begin
            mmio_we <= 1'b0;
            mmio_re <= 1'b0;
            s_awready <= 1'b0;
            s_wready <= 1'b0;
            s_arready <= 1'b0;

            // prijem adresy a dat zapisu (handshake jeden takt)
            if (s_awvalid && !aw_ok && !s_awready && !s_bvalid) begin
                s_awready <= 1'b1;
                aw_ok <= 1'b1;
                aw_q <= s_awaddr[AW-1:0];
            end
            if (s_wvalid && !w_ok && !s_wready && !s_bvalid) begin
                s_wready <= 1'b1;
                w_ok <= 1'b1;
                w_q <= s_wdata;     // HW registry jsou 32-bit, wstrb se ignoruje
            end

            if (s_bvalid && s_bready) s_bvalid <= 1'b0;
            if (s_rvalid && s_rready) s_rvalid <= 1'b0;

            // provedeni: zapis ma prednost, cteni jen kdyz neceka jine
            if (!rd_wait && aw_ok && w_ok && !s_bvalid) begin
                mmio_we <= 1'b1;
                mmio_addr <= aw_q;
                mmio_wdata <= w_q;
                aw_ok <= 1'b0;
                w_ok <= 1'b0;
                s_bvalid <= 1'b1;
            end else if (!rd_wait && s_arvalid && !s_arready && !s_rvalid && !(aw_ok && w_ok)) begin
                s_arready <= 1'b1;
                mmio_re <= 1'b1;
                mmio_addr <= s_araddr[AW-1:0];
                rd_wait <= 1'b1;
            end

            if (rd_wait && mmio_rvalid) begin
                s_rdata <= mmio_rdata;
                s_rvalid <= 1'b1;
                rd_wait <= 1'b0;
            end
        end
    end

    wire unused_ok = &{1'b0, s_wstrb};
endmodule
