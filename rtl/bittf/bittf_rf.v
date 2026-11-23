// Registrova sada BitTF mini: 64 x 64 bitu, jeden port (cteni nebo zapis), synchronni cteni (data takt po adrese).
// Behavioralni model; na ASIC (BITTF_IHP_SRAM) 4 makra IHP RM_IHPSG13_1P_64x16_c2, viz tapeout/ihp.
// dbg_* = asynchronni cteni jen pro simulaci (lockstep testbench), v syntaze nepouzito.
module bittf_rf (
    input  wire        clk,
    input  wire        en,
    input  wire        we,
    input  wire [5:0]  addr,
    input  wire [63:0] wdata,
    output wire [63:0] rdata,
    input  wire [5:0]  dbg_addr,
    output wire [63:0] dbg_rdata
);
`ifdef BITTF_IHP_SRAM
    // IHP SG13G2: 4 makra SRAM 64 x 16 (kazde 236,8 x 64,4 um), jednoportova, synchronni cteni
    RM_IHPSG13_1P_64x16_c2 sram0 (.A_CLK(clk), .A_MEN(en), .A_WEN(en & we), .A_REN(en & ~we), .A_ADDR(addr),
        .A_DIN(wdata[15:0]), .A_DLY(1'b1), .A_DOUT(rdata[15:0]));
    RM_IHPSG13_1P_64x16_c2 sram1 (.A_CLK(clk), .A_MEN(en), .A_WEN(en & we), .A_REN(en & ~we), .A_ADDR(addr),
        .A_DIN(wdata[31:16]), .A_DLY(1'b1), .A_DOUT(rdata[31:16]));
    RM_IHPSG13_1P_64x16_c2 sram2 (.A_CLK(clk), .A_MEN(en), .A_WEN(en & we), .A_REN(en & ~we), .A_ADDR(addr),
        .A_DIN(wdata[47:32]), .A_DLY(1'b1), .A_DOUT(rdata[47:32]));
    RM_IHPSG13_1P_64x16_c2 sram3 (.A_CLK(clk), .A_MEN(en), .A_WEN(en & we), .A_REN(en & ~we), .A_ADDR(addr),
        .A_DIN(wdata[63:48]), .A_DLY(1'b1), .A_DOUT(rdata[63:48]));
    assign dbg_rdata = 64'd0;
`else
    reg [63:0] mem [0:63];
    reg [63:0] rq;
    always @(posedge clk)
        if (en) begin
            if (we) mem[addr] <= wdata;
            else rq <= mem[addr];
        end
    assign rdata = rq;
    assign dbg_rdata = mem[dbg_addr];
`endif
endmodule
