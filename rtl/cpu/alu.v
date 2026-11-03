`timescale 1ns/1ps
// ALU stub
module alu(
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [3:0]  op,
    output reg [31:0] y
);
    always @(*) begin
        case(op)
            4'd0: y = a + b;
            default: y = 32'd0;
        endcase
    end
endmodule
