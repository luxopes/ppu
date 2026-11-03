`timescale 1ns/1ps
// Register file stub
module regfile(
    input wire clk,
    input wire we,
    input wire [4:0] ra1, ra2, wa,
    input wire [31:0] wd,
    output wire [31:0] rd1, rd2
);
    reg [31:0] regs [0:31];
    assign rd1 = regs[ra1];
    assign rd2 = regs[ra2];
    always @(posedge clk) if (we) regs[wa] <= wd;
endmodule
