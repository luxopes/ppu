// Registrova sada BitTF mini: 64 x 64 bitu, jeden port (cteni nebo zapis), synchronni cteni (data takt po adrese).
// Behavioralni model; na ASIC se nahradi makrem SRAM (IHP RM_IHPSG13_1P_64x64), viz tapeout/ihp.
// dbg_* = asynchronni cteni jen pro simulaci (lockstep testbench), v syntaze nepouzito.
module bittf_rf (
    input  wire        clk,
    input  wire        en,
    input  wire        we,
    input  wire [5:0]  addr,
    input  wire [63:0] wdata,
    output reg  [63:0] rdata,
    input  wire [5:0]  dbg_addr,
    output wire [63:0] dbg_rdata
);
    reg [63:0] mem [0:63];
    always @(posedge clk)
        if (en) begin
            if (we) mem[addr] <= wdata;
            else rdata <= mem[addr];
        end
    assign dbg_rdata = mem[dbg_addr];
endmodule
