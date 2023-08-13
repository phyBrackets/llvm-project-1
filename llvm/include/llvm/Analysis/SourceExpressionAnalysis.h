

#ifndef LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H
#define LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <optional>
#include <string_view>
using namespace llvm;

namespace llvm {

class LoadStoreSourceExpression {
public:
  // Constructor that takes a Function reference.
  LoadStoreSourceExpression(const Function &F) : F(F) {}

  // Print out the values currently in the cache.
  void print(raw_ostream &OS) const;

  // Query the SourceExpressionMap Using a Value
  std::string getSourceExpressionForValue(Value *key) const {
    auto it = SourceExpressionsMap.find(key);
    if (it != SourceExpressionsMap.end()) {
      return it->second;
    }

    return "Complex Expression or load and store get optimized out";
  }

  // Get the expression string corresponding to an opcode.
  std::string getExpressionFromOpcode(unsigned Opcode);

  // Process a StoreInst instruction and return its source-level expression.
  void processStoreInst(StoreInst *I);

  // Process a LoadInst instruction and update the sourceExpressionsMap.
  void processLoadInst(LoadInst *I);

private:
  // This map stores the source-level expressions for LLVM values.
  // The expressions are represented as strings and are associated with the
  // corresponding values. It is used to cache and retrieve source expressions
  // during the generation process.
  std::map<Value *, std::string> SourceExpressionsMap;

  // Process Debug Metadata associated with a stored value
  DILocalVariable *processDbgMetadata(Value *StoredValue);

  const Function &F;

  // This data structure is used to store information about the members of a
  // structure. It is implemented as a `DenseMap`, where the keys are of type
  // `StringRef` and represent the name of the base pointer or the object name,
  // and the values are vectors of pairs. Each pair consists of two strings,
  // representing the member name and the processed type information of the
  // member.

  DenseMap<StringRef, std::vector<std::pair<std::string, std::string>>>
      MemberInfo;

  // Process the DIType and store important information such as structure member
  // names
  void processDIType(DIType *TypeToBeProcessed, Value *BasePointer,
                     std::string MemberName = "");

  // Get the source-level expression for an LLVM value.
  std::string getSourceExpression(Value *Operand);

  // Get the source-level expression for a GetElementPtr instruction.
  std::string
  getSourceExpressionForGetElementPtr(GetElementPtrInst *GepInstruction);

  // Get the source-level expression for a BinaryOperator.
  std::string getSourceExpressionForBinaryOperator(BinaryOperator *BinaryOp,
                                                   Value *Operand);

  // Get the source-level expression for a LoadInst.
  std::string getSourceExpressionForLoadInst(LoadInst *LoadInstruction);

  // Get the source-level expression for a StoreInst.
  std::string getSourceExpressionForStoreInst(StoreInst *StoreInstruction);

  // Get the source-level expression for a SExtInst.
  std::string getSourceExpressionForSExtInst(SExtInst *SextInstruction);
};

class SourceExpressionAnalysis
    : public AnalysisInfoMixin<SourceExpressionAnalysis> {
  friend AnalysisInfoMixin<SourceExpressionAnalysis>;
  static AnalysisKey Key;

public:
  using Result = LoadStoreSourceExpression;
  Result run(Function &F, FunctionAnalysisManager &);
};

class SourceExpressionAnalysisPrinterPass
    : public PassInfoMixin<SourceExpressionAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit SourceExpressionAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif
