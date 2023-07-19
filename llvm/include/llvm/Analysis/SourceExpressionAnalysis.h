

#ifndef LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H
#define LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include <optional>
using namespace llvm;

namespace llvm {

class LoadStoreSourceExpression {
public:
  // Constructor that takes a Function reference.
  LoadStoreSourceExpression(const Function &F) : F(F) {}

  // Print out the values currently in the cache.
  void print(raw_ostream &OS) const;

  // Get the expression string corresponding to an opcode.
  std::string getExpressionFromOpcode(unsigned opcode);

  // Build the source-level expression for an LLVM instruction.
  void buildSourceLevelExpression(Instruction &I, StringRef symbol);

  // This map stores the source-level expressions for LLVM values.
  // The expressions are represented as strings and are associated with the
  // corresponding values. It is used to cache and retrieve source expressions
  // during the generation process.
  DenseMap<Value *, std::string> sourceExpressionsMap;

private:
  // This map associates StoreInst pointers with their corresponding LoadInst
  // pointers. It is used to track the relationship between store and load
  // instructions for later processing.
  DenseMap<StoreInst *, LoadInst *> loadStoreMap;

  const Function &F;

  // Process Debug Metadata associated with a stored value
  DILocalVariable *processDbgMetadata(Value *storedValue);

  // Remove the ampersand character from a string.
  std::string removeAmpersand(StringRef str);

  /**
   * This data structure is used to store information about the members of a
   * structure. It is implemented as a `DenseMap`, where the keys are of type
   * `StringRef` and represent the name of the base pointer or the object name,
   * and the values are vectors of pairs. Each pair consists of two strings,
   * representing the member name and the processed type information of the
   * member.
   */
  DenseMap<StringRef, std::vector<std::pair<std::string, std::string>>>
      memberInfo;

  // Check if the given DIType represents a structure type.
  bool isStructType(DIType *diType, Value *basePointer,
                    std::string memberName = "");

  //  It is used to track whether a certain array type has been encountered or
  //  not.
  std::unordered_map<std::string, bool> checkArrayType;

  // Get the source-level expression for an LLVM value.
  std::string getSourceExpression(Value *operand, StringRef symbol = "");

  // Get the source-level expression for a GetElementPtr instruction.
  std::string getSourceExpressionForGetElementPtr(GetElementPtrInst *gepInst);

  // Get the source-level expression for a BinaryOperator.
  std::string getSourceExpressionForBinaryOperator(BinaryOperator *binaryOp,
                                                   Value *operand);

  // Get the source-level expression for a LoadInst.
  std::string getSourceExpressionForLoadInst(LoadInst *loadInst);

  // Get the source-level expression for a StoreInst.
  std::string getSourceExpressionForStoreInst(StoreInst *storeInst);

  // Get the source-level expression for a SExtInst.
  std::string getSourceExpressionForSExtInst(SExtInst *sextInst);

  // Process a StoreInst instruction and return its source-level expression.
  std::string processStoreInst(StoreInst *I,

                               StringRef symbol, bool loadFlag = false);

  // Process a LoadInst instruction and update the sourceExpressionsMap.
  void processLoadInst(LoadInst *I,

                       StringRef symbol);
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
